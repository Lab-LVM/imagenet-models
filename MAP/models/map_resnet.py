from functools import partial

import torch
from timm import create_model
from torch import nn
import torch.nn.functional as F

from timm.models import register_model
from timm.models.layers import trunc_normal_

try:
    from map import MAPHead, NormHead, Head, SplitNormHead
except:
    from .map import MAPHead, NormHead, Head, SplitNormHead


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True,
                 # non_linearity=partial(nn.ReLU, inplace=True),
                 non_linearity=nn.GELU,
                 ):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            non_linearity() if act else nn.Identity()
        )


class SEUnit(nn.Sequential):
    def __init__(self, ch, norm_layer, r=16):
        super(SEUnit, self).__init__(
            nn.AdaptiveAvgPool2d(1), ConvNormAct(ch, ch // r, 1, norm_layer),
            nn.Conv2d(ch // r, ch, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        out = super(SEUnit, self).forward(x)
        return out * x


class BottleNeck(nn.Module):
    factor = 4

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = ConvNormAct(in_channels, width, 1, norm_layer)
        self.conv2 = ConvNormAct(width, width, 3, norm_layer, stride, 1, groups=groups)
        self.conv3 = ConvNormAct(width, self.out_channels, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)
        self.se = SEUnit(self.out_channels, norm_layer) if se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(self.conv3(out))
        residual = self.downsample(x)
        return self.relu(residual + self.drop_path(out))


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode='row'):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * x.new_empty(shape).bernoulli_(self.survival).div_(self.survival)


class Class_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., n_tokens=1, emb_reduction=1):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.emb_reduction = emb_reduction
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim // emb_reduction, bias=qkv_bias)
        self.k = nn.Linear(dim, dim // emb_reduction, bias=qkv_bias)
        self.v = nn.Linear(dim, dim // emb_reduction, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // emb_reduction, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.n_tokens = n_tokens

    def forward(self, x):
        cls, img = x[:, :self.n_tokens], x
        B, N, C = img.shape
        C = C // self.emb_reduction

        q = self.q(cls).reshape(B, self.n_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(img).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(img).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn_score = (q @ k.transpose(-2, -1))
        attn_org = attn_score.softmax(dim=-1)
        attn = self.attn_drop(attn_org)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.n_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def channel_shuffle(x, group):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batchsize, num_channels, height, width)

    return x


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


class CA(nn.Module):
    def __init__(self, dim, num_heads=32, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, n_tokens=1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 Attention_block=Class_Attention, Mlp_block=Mlp, emb_reduction=1, mlp_reduction=1):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, n_tokens=n_tokens, emb_reduction=emb_reduction)
        self.mlp = GroupConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, groups=mlp_reduction)
        # print(f"the number of parameters (attn): {count_parameters(self.attn) / 1000000:.2f}M")
        # print(f"the number of parameters (mlp): {count_parameters(self.mlp) / 1000000:.2f}M")
        # self.mlp = Mlp_block(in_features=dim // emb_reduction, hidden_features=mlp_hidden_dim,
        #                     act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_cls, x_img = x
        u = torch.cat((x_cls, x_img), dim=1)
        x_cls = x_cls + self.attn(self.norm1(u))
        x_cls = x_cls + self.mlp(self.norm2(x_cls))

        return x_cls, x_img


class CAP(nn.Module):
    def __init__(self, last_dim=1024, n_layers=1, n_tokens=1, distill_tokens=0, token_distill=False,
                 drop_rate=0.0, num_heads=32, emb_reduction=1, mlp_reduction=1, hf=2):
        super(CAP, self).__init__()
        all_tokens = cls_tokens = (n_tokens + distill_tokens)
        if token_distill:
            all_tokens += 1
        self.x_cls = nn.Parameter(torch.zeros([1, cls_tokens, last_dim]))
        self.attention = nn.Sequential(*[CA(last_dim, num_heads=num_heads,
                                            n_tokens=all_tokens, drop=drop_rate,
                                            attn_drop=drop_rate, emb_reduction=emb_reduction,
                                            mlp_reduction=mlp_reduction, mlp_ratio=hf) for _ in range(n_layers)])
        self.dim = int(last_dim * all_tokens)
        self.token_distill = token_distill

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x_cls = self.x_cls.expand(B, -1, -1)

        if self.token_distill:
            mean_token = self.x_cls.expand(B, -1, -1).mean(dim=1, keepdim=True)
            x_cls = torch.cat([x_cls, mean_token], dim=1)

        x_cls, x = self.attention((x_cls, x))
        return x_cls.reshape(-1, self.dim)


class VariousPool(nn.Module):
    def __init__(self, pool_types='cap', last_dim=2048, n_layers=1, n_tokens=1, distill_tokens=0,
                 drop_rate=0.0, num_heads=32, token_distill=False, emb_reduction=1, mlp_reduction=1, hf=2):
        super(VariousPool, self).__init__()
        if pool_types == 'cap':
            self.pool = CAP(last_dim, n_layers, n_tokens, distill_tokens,
                            token_distill, drop_rate, num_heads, emb_reduction=emb_reduction,
                            mlp_reduction=mlp_reduction, hf=hf)
        else:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.pool(x)


class MultiScale(nn.Module):
    def __init__(self, multi_scale_level=0, channels=[64, 256, 512, 1024, 2048], concat_conv=None):
        super(MultiScale, self).__init__()
        self.multi_scale_level = multi_scale_level
        self.use_multi_scale = multi_scale_level > 0
        self.out_dim = channels[multi_scale_level]

        if self.use_multi_scale:
            self.channels = channels
            if concat_conv is not None:
                self.concat_conv = concat_conv
            else:
                self.concat_conv = ConvNormAct(sum(channels), self.out_dim, 1)

    def forward(self, x):
        if not self.use_multi_scale:
            return x[-1]

        B, C, H, W = x[self.multi_scale_level].shape
        multi_scale_feature = []
        for feature in x:
            if H > feature.size(2):
                feature = F.adaptive_avg_pool2d(feature, (H, W))
            elif H < feature.size(2):
                feature = F.interpolate(feature, size=(H, W), mode='bilinear')
            multi_scale_feature.append(feature)

        return self.concat_conv(torch.cat(multi_scale_feature, dim=1))


class MAP_ResNet(nn.Module):
    def __init__(self,
                 nblock, block=BottleNeck, norm_layer=nn.BatchNorm2d,
                 channels=[64, 128, 256, 512], strides=[1, 2, 2, 2], groups=1, base_width=64,
                 zero_init_last=True, num_classes=1000, in_channels=3, drop_path_rate=0.0,
                 se=False, stem_type='normal', avg_down=False, dropout=0.0,
                 pool_type='cap', last_dim=384, n_groups=4, n_tokens=3, gram_group=24, ch_reduce=1,
                 token_distill=True, multi_scale_level=3, light=False, split_norm=False) -> None:
        super(MAP_ResNet, self).__init__()

        self.groups = groups
        self.num_classes = num_classes
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]
        self.num_block = sum(nblock)
        self.cur_block = 0
        self.drop_path_rate = drop_path_rate
        self.se = se
        self.avg_down = avg_down
        self.pool_type = pool_type

        if stem_type == 'deep':
            stem_chs = (in_channels, 64, 64, self.in_channels)
            self.stem = nn.Sequential(ConvNormAct(stem_chs[0], stem_chs[1], 3, self.norm_layer, 2, 1),
                                      ConvNormAct(stem_chs[1], stem_chs[2], 3, self.norm_layer, 1, 1),
                                      ConvNormAct(stem_chs[2], stem_chs[3], 3, self.norm_layer, 1, 1), )
        else:
            self.stem = ConvNormAct(in_channels, self.in_channels, 7, self.norm_layer, 2, 3)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.layers = [self.make_layer(
            block=block, nblock=nblock[i], channels=channels[i], stride=strides[i]
        ) for i in range(len(nblock))]

        if self.pool_type == 'map':
            if split_norm:
                head_fn = SplitNormHead
            else:
                head_fn = NormHead

            self.head = MAPHead(
                multi_scale_level=multi_scale_level, channels=[64] + [ch * 4 for ch in channels], last_dim=last_dim,
                n_tokens=n_tokens, n_groups=n_groups, self_distill_token=token_distill, mlp_ratio=4, mlp_groups=2,
                head_fn=head_fn, fc_drop=0, num_classes=num_classes,
                non_linearity=nn.GELU, gram=True,
                bp_dim=last_dim, bp_groups=1, gram_group=gram_group, gram_dim=last_dim,
                concat_blk=None, gram_blk=nn.Identity, ca_dim=384, num_heads=12, light=light,  # original: 192, 6
                dropout=dropout, interactive=True,
            )
        elif self.pool_type == 'multi_gap':
            class MultiGAP(nn.Module):
                def __init__(self, n_groups, num_classes):
                    super().__init__()
                    self.concat_conv = ConvNormAct(64 + 256 + 512 + 1024 + 2048, 1024, 1)
                    self.fc = nn.Linear(1024, num_classes * n_groups)
                    self.n_groups = n_groups
                    self.num_classes = num_classes

                def forward(self, x, pre_logits=False):
                    B, C, H, W = x[3].shape
                    multi_scale_feature = []
                    for feature in x:
                        if H > feature.size(2):
                            feature = F.adaptive_avg_pool2d(feature, (H, W))
                        elif H < feature.size(2):
                            feature = F.interpolate(feature, size=(H, W), mode='bilinear')
                        multi_scale_feature.append(feature)
                    multi_scale_feature = self.concat_conv(torch.cat(multi_scale_feature, dim=1))
                    logit = multi_scale_feature.mean(dim=[-1, -2])
                    logit = self.fc(logit).reshape(B, self.n_groups, self.num_classes)
                    if not pre_logits:
                        logit = logit.sum(dim=1)
                    return logit

            self.head = MultiGAP(n_groups, num_classes)
        else:
            self.head = nn.Linear(channels[0], num_classes)

        self.register_layer()
        self.init_weight(zero_init_last)

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def get_drop_path_rate(self):
        drop_path_rate = self.drop_path_rate * (self.cur_block / self.num_block)
        self.cur_block += 1
        return drop_path_rate

    def make_layer(self, block, nblock: int, channels: int, stride: int) -> nn.Sequential:
        if self.in_channels != channels * block.factor or stride != 1:
            if self.avg_down and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False),
                    ConvNormAct(self.in_channels, channels * block.factor, 1, self.norm_layer, act=False)
                )
            else:
                downsample = ConvNormAct(
                    self.in_channels, channels * block.factor, 1, self.norm_layer, stride, act=False
                )
        else:
            downsample = None

        layers = []
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                self.in_channels = channels * block.factor
            layers.append(block(self.in_channels, channels, stride, self.norm_layer, downsample,
                                self.groups, self.base_width, self.get_drop_path_rate(), self.se))
        return nn.Sequential(*layers)

    def forward(self, x, pre_logits=False):
        stem = self.stem(x)
        x = self.max_pool(stem)

        features = [stem]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features.append(x)

        if self.pool_type in ['mmcap', 'multi_gap']:
            if pre_logits:
                return self.head(features, pre_logits=True)
            else:
                return self.head(features)
        else:
            return self.head(x.mean([-2, -1]))

    def init_weight(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if zero_init_last:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.conv3[1].weight, 0)


@register_model
def map_resnet50(pretrained=False, **kwargs):
    model = MAP_ResNet(nblock=[3, 4, 6, 3], channels=[64, 128, 256, 256],
                       drop_path_rate=kwargs.get('drop_path_rate', 0.0),
                       dropout=kwargs.get('drop', 0.0),
                       num_classes=kwargs.get('num_classes', 1000),
                       pool_type='map', last_dim=384, n_groups=4, n_tokens=4, gram_group=32, ch_reduce=1,
                       se=True, stem_type='deep', token_distill=True)

    if pretrained:
        checkpoint = 'https://github.com/Lab-LVM/imagenet-models/releases/download/v0.0.1/map_resnet50.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu')
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = create_model('mmcap_384d_b4_t4_g32_1r_14ms_td_resnet50d_se_1024')
    ys = model(x)
    for y in ys:
        for item in y:
            print(item.shape)
    print('done.')
