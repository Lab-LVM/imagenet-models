""" Implementation of ga-ConvNeXt
Original code: https://github.com/facebookresearch/ConvNeXt and https://github.com/huggingface/pytorch-image-models
"""
# Copyright from the original codebase
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import register_notrace_module
from timm.models import named_apply, build_model_with_cfg
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp, create_attn
from timm.models import register_model

import numpy as np
from torch.autograd import Function

__all__ = ['GA_ConvNeXt']  # model_registry will add each entrypoint fn to this

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    ga_convnext_tiny=_cfg(url=""),
    ga_convnext_small=_cfg(url=""),
    ga_convnext_base=_cfg(url=""),
)


def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None, groups=1):
        super().__init__()

        norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)

        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = Mlp(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=2, depth=2, dp_rates=None, ls_init_value=1.0, conv_mlp=False,
            norm_layer=None, cl_norm_layer=None, groups=1, stage3_naggre=2):
        super().__init__()

        self.groups = groups
        self.stage3_naggre = stage3_naggre

        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride, groups=groups),
            )
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.Sequential(*[ConvNeXtBlock(
            dim=out_chs, drop_path=dp_rates[j], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
            norm_layer=norm_layer if conv_mlp else cl_norm_layer, groups=groups)
            for j in range(depth)]
                                    )

    def forward(self, x):
        x = self.downsample(x)
        if len(self.blocks) > 5:
            xs = []
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                if (i+1) % (len(self.blocks)//(self.stage3_naggre+1)) == 0 and len(xs) < (self.stage3_naggre):
                    xs.append(x)
            return x, xs
        else:
            x = self.blocks(x)
        return x


class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_embed=128):
        super().__init__()
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        head_dim = self.dim_embed // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, self.dim_embed , bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim_embed , bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim_embed , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_embed , dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = self.dim_embed
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class GroupConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, drop=0., groups=1):
        super().__init__()
        self.groups = groups

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True, groups=groups)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True, groups=groups)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_ndim = x.ndim
        if x_ndim == 3:
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(-1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = channel_shuffle(x, self.groups)
        x = self.fc2(x)
        if x_ndim == 3:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)
        return x


class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=GroupConvMlp, mlp_block_groups=2, init_values=1e-4, dim_embed=128):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, dim_embed=dim_embed)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim // 1, hidden_features=mlp_hidden_dim // 1, act_layer=act_layer,
                             out_features=dim // 1, drop=drop, groups=mlp_block_groups)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, outplanes, stride=1, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 aa_layer=None, drop_path=None):
        super(Bottleneck, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1),
            norm_layer(outplanes)
        )

        width = int((planes * (base_width // 64)) * cardinality)
        first_planes = width // reduce_first
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)

        self.se = create_attn('se', width, rd_ratio=1. / 4)


        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)


        self.act3 = act_layer(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.drop_path = DropPath(drop_path)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):

        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.se(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x

class GA_ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self, in_chans=3, num_classes=1000, output_stride=32, patch_size=4,
            depths=(3, 3, 9, 3, 1), dims=(96, 192, 384, 768, 768), ls_init_value=1e-6, conv_mlp=False,
            head_init_scale=1., norm_layer=None, drop_rate=0., drop_path_rate=0.,
            branches=5, gram_embedding_gropus=8, dim_embed=128, stage3_naggre=2, gram_dim=192, gram_layer=True
    ):
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp, \
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        curr_stride = patch_size
        prev_chs = dims[0]
        stages = []

        layers = len(dims)
        for i in range(layers):
            stride = 2 if i > 0 else 1
            if i == 4:
                stride = 1
                prev_chs = sum(dims[:-1]) + dims[2] * stage3_naggre
                curr_stride /= stride
                out_chs = dims[i]

                stages.append(Bottleneck(inplanes=prev_chs, planes=out_chs // 4, outplanes=out_chs, drop_path=drop_path_rate))
                prev_chs = out_chs

            else:
                # FIXME support dilation / output_stride
                curr_stride *= stride
                out_chs = dims[i]
                stages.append(ConvNeXtStage(
                    prev_chs, out_chs, stride=stride,
                    depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                    norm_layer=norm_layer, cl_norm_layer=cl_norm_layer, stage3_naggre=stage3_naggre)
                )
                prev_chs = out_chs

            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        self.branches = branches
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=14)
        self.gram_contraction = nn.ModuleList()
        self.gram_layer = nn.ModuleList()
        self.gram_layer = nn.ModuleList()
        self.gram_embedding = nn.ModuleList()

        self.ga = nn.ModuleList()  # class attention layers
        self.fc = nn.ModuleList()

        for i in range(self.branches):
            self.gram_contraction.append(nn.Sequential(nn.Conv2d(dims[-1], gram_dim, kernel_size=1, stride=1,
                                                                 padding=0, bias=True, groups=1),
                                                       nn.BatchNorm2d(gram_dim)))
            if gram_layer:
                self.gram_layer.append(ConvNeXtStage(
                    gram_dim, gram_dim, stride=1,
                    depth=1, dp_rates=dp_rates[-1], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                    norm_layer=norm_layer, cl_norm_layer=cl_norm_layer)
                )
            else:
                self.gram_layer.append(nn.Identity())
            self.gram_embedding.append(nn.Sequential(nn.Conv2d(((gram_dim + 1) * gram_dim // 2), dims[-1],
                                                               kernel_size=1, stride=1, padding=0, bias=True,
                                                               groups=gram_embedding_gropus), nn.BatchNorm2d(dims[-1])))
            self.ga.append(LayerScaleBlockClassAttn(dims[-1], num_heads=8, mlp_block_groups=4, dim_embed=dim_embed))
            self.fc.append(nn.Linear(dims[-1], num_classes))

        self.gram_index = np.zeros(((gram_dim + 1) * gram_dim // 2))
        count = 0
        for i in range(gram_dim):
            for j in range(gram_dim):
                if j >= i:
                    self.gram_index[count] = (i * gram_dim) + j
                    count += 1

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
        if isinstance(self.head, ClassifierHead):
            # norm -> global pool -> fc
            self.head = ClassifierHead(
                self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        else:
            # pool -> norm -> fc
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', self.head.norm),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))

    def get_gram(self, xbp, B, C):

        xbp = xbp/xbp.size()[2]

        if self.training and B < 128:
            xbp = xbp.to(dtype=torch.float64, memory_format=torch.contiguous_format)

        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], xbp.size()[2] * xbp.size()[3]))
        xbp = torch.bmm(xbp, torch.transpose(xbp, 1, 2)) / (xbp.size()[2])
        xbp = torch.reshape(xbp, (B, C ** 2))
        xbp = xbp[:, self.gram_index]

        xbp = torch.nn.functional.normalize(xbp)
        xbp = xbp.float()
        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], 1, 1))
        return xbp

    def forward_features(self, x):
        x = self.stem(x)
        x_cat = []
        for i in range(len(self.stages)-1):
            if i == 2:
                x, x3 = self.stages[i](x)
            else:
                x = self.stages[i](x)
            x_cat.append(x)

        x3_cat = x3[0]
        for j in range(1, len(x3)):
            x3_cat = torch.cat((x3_cat, x3[j]), dim=1)
        x = torch.cat((self.avg_pool(x_cat[0]), self.avg_pool(x_cat[1]), x3_cat, x_cat[2],
                      self.upsample(x_cat[-1])), dim=1)
        x = self.stages[-1](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x_out = []
        for k in range(self.branches):
            gram = self.gram_contraction[k](x)
            gram = self.gram_layer[k](gram)
            B, C, _, _ = gram.shape
            gram = self.get_gram(gram, B, C)
            gram = self.gram_embedding[k](gram)

            gram = gram.view(gram.size()[0], gram.size()[1], -1)
            gram = gram.permute(0, 2, 1)
            gram = self.ga[k](x.view(x.size()[0], x.size()[1], -1).permute(0, 2, 1), gram)

            gram = gram.view(gram.size(0), -1)
            gram = self.fc[k](gram)
            x_out.append(gram)
        return x_out


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict

class sign_sqrt(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input) * torch.sqrt(torch.abs(input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        grad_input = torch.div(grad_output, ((torch.abs(output) + 0.03) * 2.))
        return grad_input


def channel_shuffle(x, group):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batchsize, num_channels, height, width)

    return x


def _create_convnext(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(GA_ConvNeXt, variant, pretrained, **kwargs)

@register_model
def ga_convnext_tiny_688(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 9, 3, 1], dims=[96, 192, 384, 688, 688], gram_embedding_gropus=8, **kwargs)
    model = _create_convnext('ga_convnext_tiny', pretrained=pretrained, dim_embed=168, stage3_naggre=2, gram_dim=192, **model_args)
    return model

@register_model
def ga_convnext_tiny_768(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 9, 3, 1], dims=[96, 192, 384, 768, 768], gram_embedding_gropus=8, **kwargs)
    model = _create_convnext('ga_convnext_tiny', pretrained=pretrained, dim_embed=192, stage3_naggre=2, gram_dim=192,
                              **model_args)
    return model

@register_model
def ga_convnext_small_688(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3, 1], dims=[96, 192, 384, 688, 688], gram_embedding_gropus=8, **kwargs)
    model = _create_convnext('ga_convnext_small', pretrained=pretrained, dim_embed=168, stage3_naggre=4, gram_dim=192,
                              **model_args)
    return model

@register_model
def ga_convnext_small_768(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3, 1], dims=[96, 192, 384, 768, 768], gram_embedding_gropus=8, **kwargs)
    model = _create_convnext('ga_convnext_small', pretrained=pretrained, dim_embed=192, stage3_naggre=4, gram_dim=192,
                              **model_args)
    return model

@register_model
def ga_convnext_base_976(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3, 1], dims=[128, 256, 512, 976, 976], gram_embedding_gropus=8,
                      dim_embed=240, stage3_naggre=4, gram_dim=192,
                      **kwargs)
    model = _create_convnext('ga_convnext_base', pretrained=pretrained, **model_args)
    return model

@register_model
def ga_convnext_base_1024(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3, 1], dims=[128, 256, 512, 1024, 1024], gram_embedding_gropus=8,
                      dim_embed=256, stage3_naggre=4, gram_dim=192,
                      **kwargs)
    model = _create_convnext('ga_convnext_base', pretrained=pretrained, **model_args)
    return model
