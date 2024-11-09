import torch
import torch.nn as nn
from timm import create_model
from timm.models import register_model
from torchsummary import summary

try:
    from map import MAPHead, NormMlpHead, NormHead, SplitNormHead, Head, ConvNeXtBlk, GroupConvMlp
except:
    from .map import MAPHead, NormMlpHead, NormHead, SplitNormHead, Head, ConvNeXtBlk, GroupConvMlp


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes, use_map=False):
        super(MobileNetV1, self).__init__()
        self.num_classes = n_classes
        self.use_map = use_map

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.layers = nn.ModuleList([
            nn.Sequential(
                conv_bn(ch_in, 32, 2),
                conv_dw(32, 64, 1),
            ),
            nn.Sequential(
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
            ),
            nn.Sequential(
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
            ),
            nn.Sequential(
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
            ),
            nn.Sequential(
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
            ),
        ])
        self.use_map = use_map
        if self.use_map:
            channels = [64, 128, 256, 512, 1024]
            dim = 192
            num_heads = dim // 32
            self.norm = nn.Identity()
            self.fc = MAPHead(
                # multi-scale
                multi_scale_level=-1, channels=channels, last_dim=dim,
                # multi-token
                n_tokens=4, n_groups=1, self_distill_token=False,
                # gram
                non_linearity=nn.GELU, gram=True, concat_blk=None, gram_blk=nn.Identity,
                bp_dim=dim, bp_groups=1, gram_group=32, gram_dim=dim,
                # class attention
                num_heads=num_heads, ca_dim=dim, mlp_ratio=1, mlp_groups=1, interactive=True,
                # FC layer
                head_fn=nn.Linear, fc_drop=0, num_classes=n_classes,
            )
        else:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(1024, n_classes)
            )

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        if self.use_map:
            return self.fc(features)
        else:
            x = self.fc(x)
            return x


@register_model
def mobilenet_v1(**kwargs):
    return MobileNetV1(ch_in=3, n_classes=1000)


@register_model
def map_mobilenet_v1(pretrained=True, **kwargs):
    model = MobileNetV1(ch_in=3, n_classes=1000, use_map=True)
    if pretrained:
        checkpoint = 'https://github.com/Lab-LVM/imagenet-models/releases/download/v0.0.1/map_mobilenet_v1.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    # model check
    x = torch.rand(2, 3, 224, 224)
    model = create_model('mobilenet_v1_map')
    y = model(x)
