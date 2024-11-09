from functools import partial
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
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


def channel_shuffle(x, group: int):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batchsize, num_channels, height, width)

    return x


class GroupConvMlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., groups=1):
        super().__init__()
        self.groups = groups

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True, groups=groups)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True, groups=groups)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (b, n, c) -> (b, c, n)
        x = x.unsqueeze(-1)  # (b, c, n) -> (b, c, n, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = channel_shuffle(x, self.groups)
        x = self.fc2(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        return x


class ClassAttention(nn.Module):
    def __init__(self, in_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., n_tokens=1,
                 embed_dim=128, interactive=False, ):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim_mismatch = in_dim != dim
        self.n_tokens = n_tokens

        self.proj = nn.Linear(embed_dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.interactive = interactive

        if self.dim_mismatch:
            self.q = nn.Linear(in_dim, embed_dim, bias=qkv_bias)
            self.k1 = nn.Linear(in_dim, embed_dim, bias=qkv_bias)
            self.v1 = nn.Linear(in_dim, embed_dim, bias=qkv_bias)
            self.k2 = nn.Linear(dim, embed_dim, bias=qkv_bias)
            self.v2 = nn.Linear(dim, embed_dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, embed_dim, bias=qkv_bias)
            self.k = nn.Linear(dim, embed_dim, bias=qkv_bias)
            self.v = nn.Linear(dim, embed_dim, bias=qkv_bias)

        if self.interactive:
            self.w1 = nn.Linear(num_heads, num_heads)
            self.w2 = nn.Linear(num_heads, num_heads)

    def forward(self, x):
        if self.dim_mismatch:
            cls, img = x
            _, N1, _ = cls.shape
            B, N2, C = img.shape
            C = self.embed_dim

            q = self.q(cls).reshape(B, self.n_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q * self.scale

            k_img = self.k1(cls).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k_cls = self.k2(img).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat([k_img, k_cls], dim=-2)

            v_img = self.v1(cls).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_cls = self.v2(img).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = torch.cat([v_img, v_cls], dim=-2)
        else:
            cls, img = x[:, :self.n_tokens], x
            B, N, C = img.shape
            C = self.embed_dim

            q = self.q(cls).reshape(B, self.n_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(img).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            q = q * self.scale
            v = self.v(img).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1).contiguous())

        if self.interactive:
            attn = attn + self.w1(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        if self.interactive:
            attn = attn + self.w2(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).contiguous().reshape(B, self.n_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class CABlock(nn.Module):
    def __init__(self, in_dim, dim, num_heads=32, mlp_ratio=4.,
                 groups=2, qkv_bias=True, qk_scale=None, drop=0.05, attn_drop=0.05,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 Attention_block=ClassAttention, Mlp_block=GroupConvMlp, n_tokens=1, ca_dim=None, interactive=False):
        super().__init__()
        self.dim_mismatch = in_dim != dim
        self.norm2 = norm_layer(dim)
        self.attn = Attention_block(
            in_dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, n_tokens=n_tokens, embed_dim=ca_dim,
            interactive=interactive
        )
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop, groups=groups
        )

        if self.dim_mismatch:
            self.norm1_1 = norm_layer(in_dim)
            self.norm1_2 = norm_layer(dim)
        else:
            self.norm1 = norm_layer(dim)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x_cls, x_img = x

        if self.dim_mismatch:
            x_cls = self.norm1_1(x_cls)
            x_img = self.norm1_2(x_img)
            x_cls = self.attn((x_cls, x_img))
        else:
            u = torch.cat((x_cls, x_img), dim=1)
            x_cls = x_cls + self.attn(self.norm1(u))

        x_cls = x_cls + self.mlp(self.norm2(x_cls))

        return x_cls, x_img


class GramToken(nn.Module):
    def __init__(self, ch_dim, num_groups=8, num_tokens=1, bp_groups=1, bp_dim=192, out_dim=None, gram_blk=nn.Identity):
        super().__init__()
        triu_indices = torch.triu_indices(bp_dim, bp_dim)
        self.register_buffer('bp_index', triu_indices[0] * bp_dim + triu_indices[1])
        self.num_groups = num_groups
        self.num_tokens = num_tokens
        self.bp_dim = bp_dim
        self.gram_dim = bp_dim * (bp_dim + 1) // 2
        self.out_dim = out_dim if out_dim else ch_dim
        self.ch_reduction = nn.Sequential(
            nn.Conv2d(ch_dim, bp_dim, 1, bias=False, groups=bp_groups),
            nn.BatchNorm2d(bp_dim)
        )
        if gram_blk != nn.Identity:
            self.gram_blk = gram_blk(bp_dim, bp_dim)
        else:
            self.gram_blk = nn.Identity()
        self.bp_reduction = nn.Sequential(
            nn.Conv2d(self.gram_dim, self.out_dim * num_tokens, 1, bias=False, groups=self.num_groups),
            nn.BatchNorm2d(self.out_dim * num_tokens)
        )

    def forward(self, x):
        # 1. channel attention
        x = self.ch_reduction(x)
        x = self.gram_blk(x)

        b, c, h, w = x.shape

        x = x.reshape(b, c, h * w) / (h * w)
        attn = x @ x.transpose(-1, -2).contiguous()

        # 2. select upper tri region (+ channel shuffle)
        attn = attn.reshape(b, c * c)
        attn = attn[:, self.bp_index]
        attn = F.normalize(attn, dim=-1)

        attn = attn.reshape(b, -1, self.num_tokens, 1, 1)
        attn = attn.permute(0, 2, 1, 3, 4)
        attn = attn.reshape(b, self.gram_dim, 1, 1)

        # 3. class token from channel attention (+ channel shuffle)
        cls_tokens = self.bp_reduction(attn)
        cls_tokens = cls_tokens.reshape(b, self.out_dim, self.num_tokens)
        cls_tokens = cls_tokens.permute(0, 2, 1)

        return cls_tokens


class CAP(nn.Module):
    def __init__(self, last_dim=1024, num_heads=8, mlp_ratio=4., mlp_groups=2, n_layers=1, n_tokens=1,
                 distill_tokens=0, attn_drop=0.0, self_distill_token=False, act_layer=nn.GELU, Mlp_block=MLP,
                 gram=False, gram_group=8, bp_groups=1, gram_dim=None, bp_dim=192, gram_blk=nn.Identity,
                 ca_dim=None, interactive=False):
        super(CAP, self).__init__()
        all_tokens = cls_tokens = (n_tokens + distill_tokens)
        if self_distill_token:
            all_tokens += 1
        gram_dim = gram_dim if gram_dim else last_dim

        self.T = cls_tokens
        self.self_distill_token = self_distill_token
        self.dim = int(last_dim * all_tokens)
        self.attention = nn.Sequential(*[CABlock(gram_dim, last_dim, num_heads, mlp_ratio, mlp_groups,
                                                 act_layer=act_layer, Mlp_block=Mlp_block,
                                                 n_tokens=all_tokens, attn_drop=attn_drop, ca_dim=ca_dim,
                                                 interactive=interactive) for _ in range(n_layers)])

        self.gram = gram
        if self.gram:
            self.gram_token_extraction = GramToken(last_dim, num_groups=gram_group, num_tokens=n_tokens,
                                                   bp_groups=bp_groups, bp_dim=bp_dim, out_dim=gram_dim,
                                                   gram_blk=gram_blk)
        else:
            self.x_cls = nn.Parameter(torch.zeros([1, cls_tokens, last_dim]))

    def forward(self, x):
        if self.gram:
            x_cls = self.gram_token_extraction(x)
        else:
            x_cls = self.x_cls.expand(x.size(0), -1, -1)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.self_distill_token:
            adv_token = x_cls.mean(dim=1, keepdim=True)
            x_cls = torch.cat([x_cls, adv_token], dim=1)

        x_cls, x = self.attention((x_cls, x))
        return x_cls.reshape(-1, self.dim)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1,
                 act=True, non_linearity=partial(nn.ReLU, inplace=True)):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            non_linearity() if act else nn.Identity()
        )


class ConvNeXtBlk(nn.Module):
    def __init__(self, dim, out_dim, non_linearity=nn.GELU):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = non_linearity()
        self.pwconv2 = nn.Linear(4 * dim, out_dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class MultiScale(nn.Module):
    def __init__(self, multi_scale_level=0, channels=[64, 256, 512, 1024, 2048],
                 non_linearity=partial(nn.ReLU, inplace=True), scaled_dim=None,
                 concat_blk=partial(ConvNormAct, kernel_size=1)):
        super(MultiScale, self).__init__()
        self.multi_scale_level = multi_scale_level
        self.use_multi_scale = multi_scale_level > 0
        self.out_dim = scaled_dim if scaled_dim else channels[multi_scale_level]
        self.channels = channels
        self.concat_conv = concat_blk(sum(channels), self.out_dim, non_linearity=non_linearity)

    def forward(self, x):
        B, C, H, W = x[self.multi_scale_level].shape
        multi_scale_feature = []
        for feature in x:
            if H > feature.size(2):
                feature = F.adaptive_avg_pool2d(feature, (H, W))
            elif H < feature.size(2):
                feature = F.interpolate(feature, size=(H, W), mode='bilinear')
            multi_scale_feature.append(feature)
        multi_scale_feature = self.concat_conv(torch.cat(multi_scale_feature, dim=1))

        return multi_scale_feature


class MAP(nn.Module):
    def __init__(
            # multi-scale
            self, multi_scale_level=0, channels=[64, 256, 512, 1024, 2048], last_dim=1024,
            non_linearity=partial(nn.ReLU, inplace=True), concat_blk=partial(ConvNormAct, kernel_size=1),
            # gram
            gram=False, gram_group=16, bp_groups=1, gram_blk=nn.Identity, bp_dim=192, gram_dim=None,
            # multi-token
            num_heads=8, mlp_ratio=2, mlp_groups=1, n_layers=1, n_tokens=1, distill_tokens=0,
            self_distill_token=False, attn_drop=0., act_layer=nn.GELU, Mlp_block=MLP, ca_dim=None,
            # multi-group
            n_groups=1, interactive=False,
    ):
        super(MAP, self).__init__()
        self.mmcap = nn.ModuleList([
            CAP(last_dim, num_heads, mlp_ratio, mlp_groups, n_layers, n_tokens,
                distill_tokens, attn_drop, self_distill_token, act_layer=act_layer, Mlp_block=Mlp_block,
                gram=gram, gram_group=gram_group, bp_groups=bp_groups, gram_blk=gram_blk, bp_dim=bp_dim,
                gram_dim=gram_dim, ca_dim=ca_dim, interactive=interactive)
            for _ in range(n_groups)])
        self.use_multi_scale = multi_scale_level > 0

        if self.use_multi_scale:
            self.multi_scale = MultiScale(multi_scale_level, channels, scaled_dim=last_dim,
                                          non_linearity=non_linearity, concat_blk=concat_blk)
        elif last_dim != channels[-1]:
            self.channel_convertor = ConvNormAct(channels[-1], last_dim, 1)
        else:
            self.channel_convertor = nn.Identity()

    def forward(self, x):

        if self.use_multi_scale:
            x = self.multi_scale(x)
        else:
            x = x[-1]
            x = self.channel_convertor(x)

        output = [cap(x) for cap in self.mmcap]

        return output


class Head(nn.Module):
    def __init__(self, ch, num_classes, drop=0.0):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x, pre_logits=False):
        if pre_logits:
            return x
        x = self.dropout(x)
        x = self.head(x)
        return x


class NormHead(nn.Module):
    def __init__(self, ch, num_classes, drop=0.0, nt=1):
        super().__init__()
        self.nt = nt
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(ch)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x, pre_logits=False):
        x = self.norm(x)
        x = self.dropout(x)
        if pre_logits:
            b, c = x.shape
            x = x.reshape(b, self.nt, 1, -1)
            weight = self.head.weight.transpose(-1, -2).reshape(1, self.nt, -1, self.num_classes).repeat(b, 1, 1, 1)
            x = (x @ weight).squeeze(-2)
        else:
            x = self.head(x)
        return x


class SplitNormHead(nn.Module):
    def __init__(self, ch, num_classes, drop=0.0, nt=1):
        super().__init__()
        ch = ch // nt
        self.nt = nt
        self.norm = nn.ModuleList([])
        self.head = nn.ModuleList([])
        self.dropout = nn.Dropout(drop)
        for i in range(self.nt):
            self.norm.append(nn.LayerNorm(ch))
            self.head.append(nn.Linear(ch, num_classes))

    def forward(self, x, pre_logits=False):
        b, c = x.shape
        x = x.reshape(b, self.nt, -1)
        output = []

        for i in range(self.nt):
            split = x[:, i]
            split = self.norm[i](split)
            split = self.dropout(split)
            split = self.head[i](split)
            output.append(split)

        output = torch.stack(output, dim=1)
        output = output.sum(dim=1)

        return output


class NormMlpHead(nn.Module):
    def __init__(self, ch, num_classes, drop=0.0):
        super().__init__()
        self.norm_mlp = nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, ch), nn.Tanh())
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x, pre_logits=False):
        x = self.norm_mlp(x)
        if pre_logits:
            return x
        x = self.dropout(x)
        x = self.head(x)

        return x


class MAPHead(nn.Module):
    def __init__(self, channels=[64, 256, 512, 1024, 2048], last_dim=512, num_heads=8, multi_scale_level=3,
                 n_tokens=3, n_groups=4, self_distill_token=True, distill_tokens=0, attn_drop=0.05,
                 gram=False, gram_group=8, bp_groups=1, gram_blk=nn.Identity, bp_dim=192, gram_dim=None,
                 mlp_ratio=4, mlp_groups=2, fc_drop=0.0, num_classes=1000, head_fn=NormMlpHead,
                 act_layer=partial(nn.ReLU, inplace=True), Mlp_block=GroupConvMlp,
                 non_linearity=partial(nn.ReLU, inplace=True), concat_blk=None,
                 ca_dim=None, light=False, dropout=0.0, interactive=False):
        super().__init__()
        concat_blk = concat_blk if (concat_blk is not None) else partial(ConvNormAct, kernel_size=1)
        self.n_groups = n_groups
        self.out_ch = last_dim * n_tokens
        self.self_dt = self_distill_token
        self.light = light
        self.dropout = dropout
        self.mmcap = MAP(multi_scale_level=multi_scale_level, channels=channels, last_dim=last_dim,
                         num_heads=num_heads, n_tokens=n_tokens, n_groups=n_groups,
                         self_distill_token=self_distill_token, distill_tokens=distill_tokens,
                         attn_drop=attn_drop, mlp_ratio=mlp_ratio, mlp_groups=mlp_groups,
                         act_layer=act_layer, Mlp_block=Mlp_block, gram=gram, gram_group=gram_group,
                         bp_groups=bp_groups, gram_blk=gram_blk, bp_dim=bp_dim,
                         non_linearity=non_linearity, concat_blk=concat_blk, gram_dim=gram_dim, ca_dim=ca_dim,
                         interactive=interactive)
        try:
            self.heads = nn.ModuleList([head_fn(last_dim * n_tokens, num_classes, fc_drop, nt=n_tokens)
                                        for _ in range(n_groups)])
        except:
            self.heads = nn.ModuleList([head_fn(last_dim * n_tokens, num_classes) for _ in range(n_groups)])

        if self.self_dt:
            self.self_dt_heads = nn.ModuleList([NormHead(last_dim, num_classes, fc_drop) for _ in range(n_groups)])

    def train(self, mode: bool = True):
        if hasattr(self, 'self_dt_heads') and self.self_dt_heads is not None:
            if self.light:
                for p in self.heads.parameters():
                    p.requires_grad = False
            else:
                for p in self.self_dt_heads.parameters():
                    p.requires_grad = True
        super().train(mode)

    def eval(self):
        if hasattr(self, 'self_dt_heads') and self.self_dt_heads is not None:
            if self.light:
                for p in self.heads.parameters():
                    p.requires_grad = False
            else:
                for p in self.self_dt_heads.parameters():
                    p.requires_grad = False
        super().eval()

    def forward(self, x, pre_logits=False):
        output = []

        pools = self.mmcap(x)

        for i in range(self.n_groups):
            if self.self_dt:
                org_pool, avg_pool = pools[i][:, :self.out_ch], pools[i][:, self.out_ch:]
                if self.training:
                    org_pool = F.dropout(org_pool, self.dropout)
                    org_out = self.heads[i](org_pool)
                    avg_out = self.self_dt_heads[i](avg_pool)
                    output.append([org_out, avg_out])
                else:
                    if self.light:
                        avg_out = self.self_dt_heads[i](avg_pool)
                        output.append(avg_out)
                    else:
                        # avg_out = self.self_dt_heads[i](avg_pool)
                        # output.append(avg_out)
                        org_out = self.heads[i](org_pool)
                        output.append(org_out)
            else:
                output.append(self.heads[i](pools[i]))

        return output
