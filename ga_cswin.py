""" Implementation of ga-CSWin
Original code: https://github.com/microsoft/CSWin-Transformer
"""
# Copyright from the original codebase
# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------------


import torch
import torch.nn as nn
from timm.models import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_, create_attn
from timm.models import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    ga_CSWin_64_12211_tiny_224=_cfg(url=""),
    ga_CSWin_64_24322_small_224=_cfg(url="")
)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block_LCF(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1, padding=0)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., expansion=4, fap=False):
        super().__init__()
        self.num_heads = num_heads
        self.expansion = expansion
        head_dim = dim // num_heads
        head_dim = head_dim // expansion
        self.scale = head_dim ** -0.5
        self.fap = fap

        self.q = nn.Linear(dim, dim//expansion, bias=qkv_bias)
        self.k = nn.Linear(dim, dim//expansion, bias=qkv_bias)
        self.v = nn.Linear(dim, dim//expansion, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//expansion, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // self.expansion
        if self.fap:
            N = N - 5
            q = self.q(x[:, 5:]).unsqueeze(1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x[:, 0:5]).reshape(B, 5, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q * self.scale
            v = self.v(x[:, 0:5]).reshape(B, 5, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        else:
            q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            q = q * self.scale
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.fap:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
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
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = channel_shuffle(x, self.groups)
        x = self.fc2(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        return x


class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=GroupConvMlp, mlp_block_groups=2, init_values=1e-4, fap=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, fap=fap)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim//1, hidden_features=mlp_hidden_dim//1, act_layer=act_layer,
                             out_features=dim//1, drop=drop, groups=mlp_block_groups)
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

class GA_CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=64, depth=[2, 2, 6, 2],
                 split_size=[3, 5, 7], num_heads=12, mlp_ratio=4., mlp_ratio_stage4=4., mlp_ratio_stage5=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False, dims=[64, 128, 256, 512], stage3_naggre=4, ga_mlp_groups=4,
                 branches=5, gram_dim=192, deep_stem=False, stage5='CSwin'):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.stage3_naggre = stage3_naggre
        heads = num_heads


        if deep_stem:
            self.stage1_conv_embed = nn.Sequential(*[
                nn.Conv2d(in_chans, embed_dim, 3, stride=2, padding=1, bias=False),
                Rearrange('b c h w -> b (h w) c', h=img_size // 2, w=img_size // 2),
                nn.LayerNorm(embed_dim),
                Rearrange('b (h w) c -> b c h w', h=img_size // 2, w=img_size // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, stride=1, padding=1, bias=False),
                Rearrange('b c h w -> b (h w) c', h=img_size // 2, w=img_size // 2),
                nn.LayerNorm(embed_dim),
                Rearrange('b (h w) c -> b c h w', h=img_size // 2, w=img_size // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim, dims[0], 3, stride=2, padding=1, bias=False),
                Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
                nn.LayerNorm(dims[0])
            ])
        else:
            self.stage1_conv_embed = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], 7, 4, 2),
                Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
                nn.LayerNorm(dims[0])
            )

        curr_dim = dims[0]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(dims[0], dims[1])
        curr_dim = dims[1]
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.merge2 = Merge_Block(dims[1], dims[2])
        curr_dim = dims[2]
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(dims[2], dims[3]) #Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = dims[3]
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio_stage4,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:3]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[3])])

        aggre_dim = sum(dims) + dims[2] * stage3_naggre

        self.merge4 = None
        if stage5 == 'CSwin':
            self.stage5 = nn.Sequential(
                Rearrange(' b c h w -> b (h w) c', h=img_size // 16, w=img_size // 16),
                Merge_Block_LCF(aggre_dim, curr_dim), CSWinBlock(
                dim=curr_dim, num_heads=heads[4], reso=img_size // 16, mlp_ratio=mlp_ratio_stage5,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[4],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[-1], norm_layer=norm_layer),
                Rearrange('b (h w) c -> b c h w', h=img_size // 16, w=img_size // 16))
        elif stage5 == 'bottleneck':
            self.stage5 = Bottleneck(inplanes=aggre_dim, planes=curr_dim // 4, outplanes=curr_dim,
                                     drop_path=drop_path_rate)

        self.branches = branches
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=14)

        self.gram_contraction = nn.ModuleList()
        self.gram_layer = nn.ModuleList()
        self.gram_embedding = nn.ModuleList()
        self.gram_expansion = nn.ModuleList()

        self.ga = nn.ModuleList()  # class attention layers
        self.fc = nn.ModuleList()

        for i in range(branches):
            self.gram_contraction.append(
                nn.Sequential(
                    nn.Conv2d(curr_dim, gram_dim , kernel_size=1, stride=1,
                              padding=0, bias=True, groups=8),
                    nn.BatchNorm2d(gram_dim)))

            self.gram_layer.append(
                nn.Sequential(
                    Rearrange(' b c h w -> b (h w) c', h=img_size // 16, w=img_size // 16),
                CSWinBlock(
                    dim=gram_dim, num_heads=6, reso=img_size // 16, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[4],
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[-1], norm_layer=norm_layer),
                    Rearrange('b (h w) c -> b c h w', h=img_size // 16, w=img_size // 16),
                )
            )

            self.gram_embedding.append(nn.Sequential(
                nn.Conv2d(((gram_dim + 1) * gram_dim // 2),
                          curr_dim, kernel_size=1,
                          stride=1, padding=0,
                          bias=True, groups=8),
                nn.BatchNorm2d(curr_dim)))

            self.ga.append(LayerScaleBlockClassAttn(curr_dim, num_heads=8, mlp_block_groups=ga_mlp_groups, fap=False))
            self.fc.append(nn.Linear(curr_dim, num_classes))

        self.gram_index = np.zeros(((gram_dim + 1) * gram_dim // 2))
        count = 0
        for i in range(gram_dim):
            for j in range(gram_dim):
                if j >= i:
                    self.gram_index[count] = (i * gram_dim) + j
                    count += 1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def get_gram(self, xbp, B, C):
        xbp = xbp / xbp.size()[2]
        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], xbp.size()[2] * xbp.size()[3]))
        xbp = torch.bmm(xbp, torch.transpose(xbp, 1, 2)) / (xbp.size()[2])
        xbp = torch.reshape(xbp, (B, C ** 2))
        xbp = xbp[:, self.gram_index]

        xbp = torch.nn.functional.normalize(xbp)
        xbp = xbp.float()
        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], 1, 1))
        return xbp

    def forward_features(self, x):
        x = self.stage1_conv_embed(x)
        xs = []
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        B, N, C = x.shape

        xs.append(x.transpose(-2,-1).reshape(B, C, int(N**0.5), int(N**0.5)))
        l_index = 0

        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            b_index = 0
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
                b_index = b_index + 1
                if (l_index == 1) and ((b_index+1) % (len(blocks)//(self.stage3_naggre+1)) == 0 and len(xs) < (self.stage3_naggre + 2)):
                    B, N, C = x.shape
                    xs.append(x.transpose(-2, -1).reshape(B, C, int(N ** 0.5), int(N ** 0.5)))
            B, N, C = x.shape
            xs.append(x.transpose(-2, -1).reshape(B, C, int(N ** 0.5), int(N ** 0.5)))
            l_index = l_index + 1

        x = torch.cat((self.avg_pool(xs[0]), self.avg_pool(xs[1])), dim=1)
        for j in range(2, len(xs)-1):
            x = torch.cat((x, xs[j]), dim=1)
        x = torch.cat((x, self.upsample(xs[-1])), dim=1)
        x = self.stage5(x)
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

def channel_shuffle(x, group):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batchsize, num_channels, height, width)

    return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def _create_cswin(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(GA_CSWinTransformer, variant, pretrained, **kwargs)
