import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

try:
    from map import MAPHead, NormHead, Head, SplitNormHead
except:
    from .map import MAPHead, NormHead, Head, SplitNormHead


class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 global_pool='avg', last_dim=384, n_groups=4, n_tokens=3, gram_group=8, ch_reduce=1,
                 bp_dim=192, bp_groups=1, gram_layer=None, gram_dim=None,
                 ca_dim=128, num_heads=8,
                 gram=True, split_norm=False, self_distill_token=True,
                 **kwargs,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.global_pool = global_pool
        if self.global_pool == 'mmcap':
            self.norm = nn.Identity()
            if split_norm:
                head_fn = SplitNormHead
            else:
                head_fn = NormHead

            self.head = MAPHead(
                multi_scale_level=3, channels=[dims[0]] + dims, last_dim=last_dim,
                n_tokens=n_tokens, n_groups=n_groups, self_distill_token=self_distill_token, mlp_ratio=4, mlp_groups=2,
                head_fn=head_fn, fc_drop=0, num_classes=num_classes,
                non_linearity=nn.GELU, gram=gram,
                bp_dim=bp_dim, bp_groups=bp_groups, gram_group=gram_group, gram_dim=gram_dim,
                concat_blk=None, gram_blk=nn.Identity, ca_dim=ca_dim, num_heads=num_heads,
            )
        else:
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []

        for i in range(4):
            x = self.downsample_layers[i](x)
            if i == 0:
                features.append(x)
            x = self.stages[i](x)
            features.append(x)

        if self.global_pool == 'mmcap':
            return features
        else:
            return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def map_convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], global_pool='mmcap',
                     last_dim=384, n_groups=4, n_tokens=2, gram_group=24,
                     bp_dim=384, ca_dim=384, num_heads=12,
                     **kwargs)
    if pretrained:
        checkpoint = 'https://github.com/Lab-LVM/imagenet-models/releases/download/v0.0.1/map_convnext_tiny.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu')
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        model.load_state_dict(state_dict)
    return model


@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def map_convnext_small(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], global_pool='mmcap',
                     last_dim=384, n_groups=4, n_tokens=3, gram_group=16,
                     bp_dim=384, ca_dim=384, num_heads=12,
                     **kwargs)
    if pretrained:
        checkpoint = 'https://github.com/Lab-LVM/imagenet-models/releases/download/v0.0.1/map_convnext_small.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu')
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        model.load_state_dict(state_dict)
    return model
