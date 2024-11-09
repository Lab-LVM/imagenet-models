# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import torch
from einops import rearrange
from timm import create_model
from torch import nn
import torch.nn.functional as F
import math

from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block as transformer_block
from timm.models.registry import register_model
from torch.hub import load_state_dict_from_url

try:
    from map import MAPHead, NormHead, Head
except:
    from .map import MAPHead, NormHead, Head


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            transformer_block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                # drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        for blk in self.blocks:
            x = blk(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):
        x = self.conv(x)

        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0,
                 pool_type='cap', last_dim=384, n_groups=4, n_tokens=3, gram_group=24, self_distill_token=True,
                 gram=True, multi_scale_level=2,
                 ):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )
        self.embed_dim = base_dims[-1] * heads[-1]

        # changes for MAP ###############################################
        channels = [base_dims[0] * heads[0]] + [dim * head for dim, head in zip(base_dims, heads)]
        self.pool_type = pool_type
        if self.pool_type == 'map':
            self.head = MAPHead(
                multi_scale_level=multi_scale_level, channels=channels, last_dim=last_dim,
                n_tokens=n_tokens, n_groups=n_groups, self_distill_token=self_distill_token, mlp_ratio=4, mlp_groups=2,
                head_fn=NormHead, fc_drop=0, num_classes=num_classes,
                non_linearity=nn.GELU, gram=gram,
                bp_dim=last_dim, bp_groups=1, gram_group=gram_group, gram_dim=last_dim,
                concat_blk=None, gram_blk=nn.Identity, ca_dim=192, num_heads=12, light=False,
            )
        else:
            self.head = nn.Linear(channels[-1], num_classes)
        #################################################################

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        features = list()
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        features.append(x)

        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            features.append(x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        features.append(x)

        return features

    def forward_head(self, x):
        if self.pool_type == 'map':
            return self.head(x)
        else:
            return self.head(x[-1].mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def pit_s(pretrained, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        pool_type='gap',
        **kwargs
    )

    return model


@register_model
def map_pit_s(pretrained, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        pool_type='map',
        last_dim=384,
        n_groups=2,
        n_tokens=4,
        gram_group=32,
        **kwargs
    )

    if pretrained:
        checkpoint = 'https://github.com/Lab-LVM/imagenet-models/releases/download/v0.0.1/map_pit_s.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = create_model('map_pit_s')
    ys = model(x)
    for y in ys:
        for item in y:
            print(item.shape)
