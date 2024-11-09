# [MAP] Enriching Local Patterns with Multi-Token Attention for Broad-Sight Neural Networks

This folder contains official pyTorch implementations for *"Enriching Local Patterns with Multi-Token Attention for Broad-Sight Neural Networks"* accepted in WACV'25. (see our [paper]()).



## 1. Requirement

Please install PyTorch (>=1.11.0), TorchVision (0.12.0), and TIMM (0.9.2) library. 

## 2. How to use MAP?

**Usage.** You can download the code of [map.py](models/map.py) and replace the original (last pooling + FC) layer with the MAP layer. We provide short code snippets to visualize the usage of the proposed MAP in custom networks, which is extracted from [map_mobilenet.py](models/map_mobilenet.py). We set hyper-parameter `num_groups=2` and `num_tokens=4`, which generally works well in diverse networks.

```python
class MobileNetV1(nn.Module):
    def __init__(self, ch_in=3, n_classes=1000, num_groups=2, num_tokens=4):
        # load needed modules...
        
        # load our MAP module
        channels = [64, 128, 256, 512, 1024] # the number channels for 5 stage features maps
        dim = 192 # the hidden channel for MAP layer.
        self.fc = MAPHead(
            # multi-scale
            multi_scale_level=-1, channels=channels, last_dim=dim,
            # multi-token
            n_tokens=num_tokens, n_groups=num_groups, self_distill_token=False,
            # gram
            non_linearity=nn.GELU, gram=True, concat_blk=None, gram_blk=nn.Identity,
            bp_dim=dim, bp_groups=1, gram_group=32, gram_dim=dim,
            # class attention
            num_heads=dim // 32, ca_dim=dim, mlp_ratio=1, mlp_groups=1, interactive=True,
            # FC layer
            head_fn=nn.Linear, fc_drop=0, num_classes=n_classes,
        )
    def forward(self, x):
        features = []
		for layer in self.layers:
            x = layer(x)
            features.append(x)

		return self.fc(features)
   
if __name__ == '__main__':
    model = MobileNetV1(num_groups=2, num_tokens=4)
    x = torch.rand(2, 3, 224, 224)
    y = model(x) 
    print(y.shape) # [torch.tensor(2, 1000), torch.tensor(2, 1000)]
```



## 3. How to use MAP with SOTA Networks?

**Usage.** You can download the code of [map_convnext.py](models/map_convnext.py), [map_maxvit.py](models/map_maxvit.py), [map_pit.py](models/map_pit.py), [map_resnet.py](models/map_resnet.py), and [map_mobilenet.py](models/map_mobilenet.py). By importing downloaded model file, you can use pre-trained backbone network to predict images. You can choose model name from `map_mobilenet_v1`, `map_resnet50`, `map_pit_s`, `map_convnext_tiny`, `map_convnext_small`, `map_maxvit_tiny_tf_224`, and `map_faster_vit_3_224`.

```python
#!wget https://github.com/Lab-LVM/imagenet-models/blob/main/MAP/models/map_convnext.py
#!wget https://user-images.githubusercontent.com/31476895/167238573-b0cc3a6d-d3ee-462b-8630-a8f253e69bb2.png

from map_convnext import *
from timm import create_model
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

def to_tensor(img):
    transform_fn = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform_fn(img)

img = Image.open('167238573-b0cc3a6d-d3ee-462b-8630-a8f253e69bb2.png').convert('RGB')
x = to_tensor(img)
x = x.unsqueeze(0)

model = create_model('map_convnext_tiny', num_classes=1000, pretrained=True) 
y = model(x).argmax(dim=1)

print(y)

##########################################
# output (answer: 196, Sussex_spaniel)
196
##########################################
```





**Checkpoint.** We provide checkpoints of various backbone network with the proposed MAP layer. 

| Network       | Speed (img/sec) | Param (M) | FLOPs (G) | Top-1 Acc. (%) | Δ    | Link                                      |
| ------------- | --------------- | --------- | --------- | -------------- | ---- | ----------------------------------------- |
| MobileNet     | 4066            | 4.2       | 0.6       | 71.3           |      |                                           |
| +MAP (Ours)   | 3734            | 4.9       | 0.6       | 73.4           | +2.1 | [ckpt](), [code](models/map_mobilenet.py) |
| ResNet50      | 3334            | 25.6      | 4.1       | 79.8           |      |                                           |
| +MAP* (Ours)  | 2127            | 42.7      | 5.4       | 82.9           | +3.1 | [ckpt](), [code](models/map_resnet.py)    |
| PiT-S         | 2580            | 23.5      | 2.4       | 80.9           |      |                                           |
| +MAP (Ours)   | 2254            | 36.2      | 2.6       | 81.9           | +1.0 | [ckpt](), [code](models/map_pit.py)       |
| ConvNeXt-T    | 2040            | 29.0      | 4.5       | 82.1           |      |                                           |
| +MAP (Ours)   | 1665            | 47.8      | 4.9       | 83.3           | +1.2 | [ckpt](), [code](models/map_convnext.py)  |
| ConvNeXt-S    | 1257            | 50.0      | 8.7       | 83.1           |      |                                           |
| +MAP (Ours)   | 1111            | 82.8      | 9.2       | 84.1           | +1.0 | [ckpt](), [code](models/map_convnext.py)  |
| MaxViT-T      | 1009            | 30.9      | 5.4       | 83.6           |      |                                           |
| +MAP (Ours)   | 907             | 50.0      | 5.8       | 84.3           | +0.7 | [ckpt](), [code](models/map_maxvit.py)    |
| FasterViT-3** | 1087            | 159.5     | 18.5      | 83.1           |      |                                           |
| +MAP (Ours)   | 970             | 187.0     | 18.8      | 84.2           | +1.1 | [ckpt](), [code](models/map_fastervit.py) |

In this table, we evaluate the top-1 accuracy (%) on the ImageNet-1K dataset with 300 epochs. We measure the speed of networks on the RTX 3090 GPU. *: we include SE unit and deep stem in backbone to improve the performance during rebuttal phase. **: we reproduce baseline results without self-distillation method.



**Command Line.** We provide bash command line to train/validate different backbone network with the proposed MAP layer. Moreover, we denote training time and evaluation results when running each command.



<details>
    <summary>MobileNet-V1</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py mobilenet_v1 -c 0,1 -m map_mobilenet_v1
```



**Validation Command Line.** Running this command line will result in: *Acc@1 73.430 (26.570) Acc@5 91.364 (8.636)*

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_mobilenet_v1 --checkpoint ../output/result/map_mobilenet_v1.pth.tar --crop-pct 0.95

Validating in mixed precision with native PyTorch AMP.
Loaded state_dict from checkpoint '../output/result/map_mobilenet_v1.pth.tar'
Model map_mobilenet_v1 created, param count: 4879612
Data processing configuration for current model + dataset:
        input_size: (3, 224, 224)
        interpolation: bicubic
        mean: (0.485, 0.456, 0.406)
        std: (0.229, 0.224, 0.225)
        crop_pct: 0.95
        crop_mode: center
Test: [   0/196]  Time: 2.181s (2.181s,  117.40/s)  Loss:  0.5522 (0.5522)  Acc@1:  85.938 ( 85.938)  Acc@5:  97.266 ( 97.266)
Test: [  10/196]  Time: 0.031s (0.468s,  546.69/s)  Loss:  1.0576 (0.7478)  Acc@1:  78.125 ( 81.499)  Acc@5:  91.406 ( 94.922)
Test: [  20/196]  Time: 0.031s (0.435s,  588.22/s)  Loss:  0.8672 (0.7753)  Acc@1:  82.031 ( 80.748)  Acc@5:  91.406 ( 94.438)
Test: [  30/196]  Time: 0.031s (0.388s,  660.54/s)  Loss:  0.8032 (0.7351)  Acc@1:  78.906 ( 81.704)  Acc@5:  95.703 ( 94.645)
Test: [  40/196]  Time: 0.555s (0.398s,  643.40/s)  Loss:  0.8394 (0.8033)  Acc@1:  79.688 ( 79.449)  Acc@5:  94.922 ( 94.484)
Test: [  50/196]  Time: 0.031s (0.374s,  684.70/s)  Loss:  0.5137 (0.8053)  Acc@1:  88.672 ( 79.159)  Acc@5:  95.703 ( 94.669)
Test: [  60/196]  Time: 1.390s (0.394s,  650.34/s)  Loss:  1.0215 (0.8163)  Acc@1:  71.094 ( 78.855)  Acc@5:  93.359 ( 94.743)
Test: [  70/196]  Time: 0.031s (0.376s,  681.73/s)  Loss:  0.8979 (0.7989)  Acc@1:  76.172 ( 79.363)  Acc@5:  92.578 ( 94.883)
Test: [  80/196]  Time: 1.031s (0.383s,  668.91/s)  Loss:  1.4590 (0.8268)  Acc@1:  64.844 ( 78.834)  Acc@5:  89.062 ( 94.541)
Test: [  90/196]  Time: 0.032s (0.372s,  689.01/s)  Loss:  2.0449 (0.8793)  Acc@1:  48.828 ( 77.704)  Acc@5:  81.250 ( 93.879)
Test: [ 100/196]  Time: 1.321s (0.376s,  680.96/s)  Loss:  1.4414 (0.9332)  Acc@1:  60.547 ( 76.613)  Acc@5:  87.500 ( 93.232)
Test: [ 110/196]  Time: 0.031s (0.368s,  696.45/s)  Loss:  0.9023 (0.9504)  Acc@1:  76.953 ( 76.270)  Acc@5:  92.578 ( 93.014)
Test: [ 120/196]  Time: 0.525s (0.369s,  692.85/s)  Loss:  1.5664 (0.9697)  Acc@1:  64.844 ( 75.988)  Acc@5:  82.812 ( 92.691)
Test: [ 130/196]  Time: 0.031s (0.367s,  698.08/s)  Loss:  0.8574 (1.0029)  Acc@1:  76.172 ( 75.057)  Acc@5:  94.531 ( 92.357)
Test: [ 140/196]  Time: 0.217s (0.364s,  702.99/s)  Loss:  1.0977 (1.0188)  Acc@1:  71.484 ( 74.670)  Acc@5:  91.016 ( 92.160)
Test: [ 150/196]  Time: 0.031s (0.365s,  701.51/s)  Loss:  1.1875 (1.0396)  Acc@1:  74.219 ( 74.252)  Acc@5:  87.500 ( 91.828)
Test: [ 160/196]  Time: 0.091s (0.362s,  707.63/s)  Loss:  0.8408 (1.0530)  Acc@1:  79.688 ( 74.017)  Acc@5:  94.531 ( 91.627)
Test: [ 170/196]  Time: 0.032s (0.364s,  703.98/s)  Loss:  0.6958 (1.0715)  Acc@1:  82.812 ( 73.563)  Acc@5:  96.484 ( 91.422)
Test: [ 180/196]  Time: 0.031s (0.359s,  712.99/s)  Loss:  1.2354 (1.0867)  Acc@1:  66.797 ( 73.235)  Acc@5:  92.188 ( 91.203)
Test: [ 190/196]  Time: 0.031s (0.361s,  708.47/s)  Loss:  1.2754 (1.0837)  Acc@1:  67.188 ( 73.272)  Acc@5:  93.750 ( 91.296)
 * Acc@1 73.430 (26.570) Acc@5 91.364 (8.636)
--result
{
    "model": "map_mobilenet_v1",
    "top1": 73.43,
    "top1_err": 26.57,
    "top5": 91.364,
    "top5_err": 8.636,
    "param_count": 4.88,
    "img_size": 224,
    "cropt_pct": 0.95,
    "interpolation": "bicubic"
}
```



</details>



<details>
    <summary>ResNet50</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py resnet50 -c 0,1 -m map_resnet50
```



**Validation Command Line.** Running this command line will result in: *Acc@1 82.850 (17.150) Acc@5 95.946 (4.054)*

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_resnet50 --checkpoint ../output/result/map_resnet50.pth.tar --crop-pct 0.95

Validating in mixed precision with native PyTorch AMP.
Loaded state_dict from checkpoint '../output/result/map_resnet50.pth.tar'
Model map_resnet50 created, param count: 42708288
Data processing configuration for current model + dataset:
        input_size: (3, 224, 224)
        interpolation: bicubic
        mean: (0.485, 0.456, 0.406)
        std: (0.229, 0.224, 0.225)
        crop_pct: 0.95
        crop_mode: center
Test: [   0/196]  Time: 2.147s (2.147s,  119.21/s)  Loss:  0.3633 (0.3633)  Acc@1:  93.359 ( 93.359)  Acc@5:  98.438 ( 98.438)
Test: [  10/196]  Time: 0.141s (0.497s,  515.53/s)  Loss:  0.7866 (0.5349)  Acc@1:  81.641 ( 87.322)  Acc@5:  96.094 ( 97.940)
Test: [  20/196]  Time: 0.140s (0.448s,  571.50/s)  Loss:  0.4805 (0.5474)  Acc@1:  92.188 ( 87.109)  Acc@5:  96.094 ( 97.489)
Test: [  30/196]  Time: 0.141s (0.401s,  638.08/s)  Loss:  0.6211 (0.5105)  Acc@1:  86.719 ( 88.281)  Acc@5:  95.703 ( 97.606)
Test: [  40/196]  Time: 0.402s (0.403s,  635.01/s)  Loss:  0.5420 (0.5530)  Acc@1:  89.453 ( 87.309)  Acc@5:  97.656 ( 97.513)
Test: [  50/196]  Time: 0.140s (0.383s,  668.81/s)  Loss:  0.3384 (0.5546)  Acc@1:  94.531 ( 87.217)  Acc@5:  98.047 ( 97.534)
Test: [  60/196]  Time: 1.115s (0.399s,  641.80/s)  Loss:  0.7329 (0.5745)  Acc@1:  82.812 ( 86.834)  Acc@5:  96.484 ( 97.496)
Test: [  70/196]  Time: 0.140s (0.384s,  666.24/s)  Loss:  0.5967 (0.5592)  Acc@1:  86.328 ( 87.181)  Acc@5:  98.438 ( 97.629)
Test: [  80/196]  Time: 0.692s (0.389s,  657.70/s)  Loss:  1.0684 (0.5800)  Acc@1:  74.609 ( 86.796)  Acc@5:  95.703 ( 97.454)
Test: [  90/196]  Time: 0.141s (0.380s,  673.61/s)  Loss:  1.4502 (0.6135)  Acc@1:  64.844 ( 85.955)  Acc@5:  91.406 ( 97.154)
Test: [ 100/196]  Time: 0.793s (0.382s,  670.41/s)  Loss:  0.9629 (0.6526)  Acc@1:  76.562 ( 85.044)  Acc@5:  92.578 ( 96.790)
Test: [ 110/196]  Time: 0.141s (0.376s,  680.75/s)  Loss:  0.5264 (0.6631)  Acc@1:  86.719 ( 84.783)  Acc@5:  97.656 ( 96.653)
Test: [ 120/196]  Time: 0.141s (0.377s,  678.49/s)  Loss:  0.9111 (0.6704)  Acc@1:  80.469 ( 84.740)  Acc@5:  93.359 ( 96.501)
Test: [ 130/196]  Time: 0.141s (0.377s,  679.82/s)  Loss:  0.4512 (0.6929)  Acc@1:  89.844 ( 84.041)  Acc@5:  98.438 ( 96.371)
Test: [ 140/196]  Time: 0.142s (0.374s,  684.15/s)  Loss:  0.6763 (0.7028)  Acc@1:  84.375 ( 83.818)  Acc@5:  96.875 ( 96.304)
Test: [ 150/196]  Time: 0.141s (0.374s,  683.60/s)  Loss:  0.6851 (0.7135)  Acc@1:  87.109 ( 83.589)  Acc@5:  95.703 ( 96.189)
Test: [ 160/196]  Time: 0.141s (0.372s,  688.16/s)  Loss:  0.4453 (0.7222)  Acc@1:  91.016 ( 83.424)  Acc@5:  97.266 ( 96.086)
Test: [ 170/196]  Time: 0.142s (0.373s,  686.19/s)  Loss:  0.4734 (0.7341)  Acc@1:  89.844 ( 83.135)  Acc@5:  98.828 ( 95.970)
Test: [ 180/196]  Time: 0.142s (0.370s,  691.11/s)  Loss:  0.9429 (0.7444)  Acc@1:  74.609 ( 82.856)  Acc@5:  96.094 ( 95.902)
Test: [ 190/196]  Time: 0.141s (0.370s,  691.18/s)  Loss:  0.9922 (0.7462)  Acc@1:  74.609 ( 82.782)  Acc@5:  96.484 ( 95.910)
 * Acc@1 82.850 (17.150) Acc@5 95.946 (4.054)
--result
{
    "model": "map_resnet50",
    "top1": 82.85,
    "top1_err": 17.15,
    "top5": 95.946,
    "top5_err": 4.054,
    "param_count": 42.71,
    "img_size": 224,
    "cropt_pct": 0.95,
    "interpolation": "bicubic"
}
```



</details>





<details>
    <summary>PiT-S</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py pit_s -c 0,1 -m map_pit_s
```



**Validation Command Line.** Running this command line will result in: *Acc@1 81.888 (18.112) Acc@5 95.810 (4.190)*

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_pit_s --checkpoint ../output/result/map_pit_s.pth.tar --crop-pct 0.95

Validating in mixed precision with native PyTorch AMP.
Loaded state_dict from checkpoint '../output/result/map_pit_s.pth.tar'
Model map_pit_s created, param count: 36147424
Data processing configuration for current model + dataset:
        input_size: (3, 224, 224)                                                                           		interpolation: bicubic
        mean: (0.485, 0.456, 0.406)
        std: (0.229, 0.224, 0.225)
        crop_pct: 0.95
        crop_mode: center
Test: [   0/196]  Time: 2.017s (2.017s,  126.91/s)  Loss:  0.6191 (0.6191)  Acc@1:  91.406 ( 91.406)  Acc@5:  98.438 ( 98.438)
Test: [  10/196]  Time: 0.073s (0.461s,  555.87/s)  Loss:  0.9756 (0.7690)  Acc@1:  80.469 ( 86.932)  Acc@5:  95.312 ( 97.230)
Test: [  20/196]  Time: 0.073s (0.432s,  592.55/s)  Loss:  0.7188 (0.7809)  Acc@1:  93.359 ( 86.514)  Acc@5:  95.703 ( 97.210)
Test: [  30/196]  Time: 0.073s (0.386s,  663.68/s)  Loss:  0.8765 (0.7546)  Acc@1:  85.547 ( 87.550)  Acc@5:  95.312 ( 97.303)
Test: [  40/196]  Time: 0.363s (0.392s,  653.07/s)  Loss:  0.8472 (0.7992)  Acc@1:  87.891 ( 86.423)  Acc@5:  97.266 ( 97.161)
Test: [  50/196]  Time: 0.073s (0.370s,  691.94/s)  Loss:  0.6313 (0.8056)  Acc@1:  92.188 ( 86.144)  Acc@5:  98.047 ( 97.273)
Test: [  60/196]  Time: 1.071s (0.389s,  658.59/s)  Loss:  0.9331 (0.8177)  Acc@1:  82.812 ( 85.873)  Acc@5:  96.094 ( 97.323)
Test: [  70/196]  Time: 0.073s (0.373s,  686.83/s)  Loss:  0.8252 (0.8034)  Acc@1:  84.766 ( 86.163)  Acc@5:  99.219 ( 97.464)
Test: [  80/196]  Time: 0.578s (0.379s,  675.00/s)  Loss:  1.2754 (0.8180)  Acc@1:  68.750 ( 85.745)  Acc@5:  94.141 ( 97.314)
Test: [  90/196]  Time: 0.073s (0.369s,  693.59/s)  Loss:  1.5830 (0.8485)  Acc@1:  62.109 ( 84.856)  Acc@5:  91.016 ( 96.978)
Test: [ 100/196]  Time: 0.701s (0.372s,  687.29/s)  Loss:  1.1084 (0.8816)  Acc@1:  76.953 ( 83.942)  Acc@5:  94.922 ( 96.670)
Test: [ 110/196]  Time: 0.073s (0.365s,  701.08/s)  Loss:  0.8379 (0.8912)  Acc@1:  83.594 ( 83.731)  Acc@5:  97.266 ( 96.586)
Test: [ 120/196]  Time: 0.073s (0.369s,  694.34/s)  Loss:  1.1680 (0.8976)  Acc@1:  76.562 ( 83.694)  Acc@5:  94.141 ( 96.459)
Test: [ 130/196]  Time: 0.073s (0.365s,  700.74/s)  Loss:  0.6709 (0.9176)  Acc@1:  88.281 ( 83.015)  Acc@5:  98.828 ( 96.311)
Test: [ 140/196]  Time: 0.073s (0.365s,  700.86/s)  Loss:  0.9141 (0.9267)  Acc@1:  85.156 ( 82.865)  Acc@5:  95.703 ( 96.207)
Test: [ 150/196]  Time: 0.073s (0.364s,  703.82/s)  Loss:  1.0039 (0.9382)  Acc@1:  82.422 ( 82.580)  Acc@5:  94.922 ( 96.070)
Test: [ 160/196]  Time: 0.074s (0.363s,  705.85/s)  Loss:  0.7676 (0.9470)  Acc@1:  89.844 ( 82.388)  Acc@5:  96.094 ( 95.926)
Test: [ 170/196]  Time: 0.074s (0.362s,  706.40/s)  Loss:  0.6899 (0.9570)  Acc@1:  87.891 ( 82.111)  Acc@5:  98.828 ( 95.836)
Test: [ 180/196]  Time: 0.073s (0.362s,  707.86/s)  Loss:  1.0742 (0.9658)  Acc@1:  76.562 ( 81.848)  Acc@5:  97.266 ( 95.755)
Test: [ 190/196]  Time: 0.073s (0.360s,  711.28/s)  Loss:  1.0967 (0.9649)  Acc@1:  75.391 ( 81.802)  Acc@5:  98.047 ( 95.783)
 * Acc@1 81.888 (18.112) Acc@5 95.810 (4.190)
--result
{
    "model": "map_pit_s",
    "top1": 81.888,
    "top1_err": 18.112,
    "top5": 95.81,
    "top5_err": 4.19,
    "param_count": 36.15,
    "img_size": 224,
    "cropt_pct": 0.95,
    "interpolation": "bicubic"
}
```



</details>



<details>
    <summary>ConvNeXt-T</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py convnext_tiny -c 0,1 -m map_convnext_tiny
```



**Validation Command Line.** Running this command line will result in: *Acc@1 83.166 (16.834) Acc@5 96.272 (3.728)*. This result is slightly different than original table. This is because we save a checkpoint at the last epoch. We apologize for providing a checkpoint saved at the last epoch, not best epoch.

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_convnext_tiny --checkpoint ../output/result/map_convnext_tiny.pth.tar --crop-pct 0.875

Validating in mixed precision with native PyTorch AMP.                                                       Loaded state_dict from checkpoint '../output/result/map_convnext_tiny.pth.tar'                               Model map_convnext_tiny created, param count: 47833760                                                       Data processing configuration for current model + dataset:                                                   		input_size: (3, 224, 224)                                                                                   interpolation: bicubic                                                                                       mean: (0.485, 0.456, 0.406)                                                                                 std: (0.229, 0.224, 0.225)                                                                                   crop_pct: 0.875                                                                                             crop_mode: center                                                                                    Test: [   0/196]  Time: 2.565s (2.565s,   99.81/s)  Loss:  0.4282 (0.4282)  Acc@1:  92.969 ( 92.969)  Acc@5:  98.047 ( 98.047)                                                             
Test: [  10/196]  Time: 0.158s (0.544s,  470.83/s)  Loss:  0.7935 (0.5830)  Acc@1:  80.859 ( 87.500)  Acc@5:  96.875 ( 98.011)                                                             
Test: [  20/196]  Time: 0.158s (0.474s,  540.13/s)  Loss:  0.4712 (0.5862)  Acc@1:  92.578 ( 87.370)  Acc@5:  97.656 ( 97.805)                                                             
Test: [  30/196]  Time: 0.158s (0.422s,  607.18/s)  Loss:  0.6553 (0.5506)  Acc@1:  86.719 ( 88.521)  Acc@5:  97.266 ( 97.946)                                                             
Test: [  40/196]  Time: 0.158s (0.412s,  621.84/s)  Loss:  0.5591 (0.5937)  Acc@1:  88.672 ( 87.443)  Acc@5:  98.047 ( 97.723)                                                             
Test: [  50/196]  Time: 0.158s (0.391s,  654.33/s)  Loss:  0.4050 (0.5972)  Acc@1:  92.969 ( 87.240)  Acc@5:  98.047 ( 97.741)                                                             
Test: [  60/196]  Time: 0.943s (0.407s,  628.50/s)  Loss:  0.7285 (0.6144)  Acc@1:  82.812 ( 86.808)  Acc@5:  96.875 ( 97.733)                                                             
Test: [  70/196]  Time: 0.158s (0.392s,  652.42/s)  Loss:  0.6626 (0.6009)  Acc@1:  84.375 ( 87.082)  Acc@5:  98.047 ( 97.838)                                                             
Test: [  80/196]  Time: 0.578s (0.397s,  644.21/s)  Loss:  1.0674 (0.6206)  Acc@1:  72.266 ( 86.661)  Acc@5:  95.312 ( 97.666)                                                             
Test: [  90/196]  Time: 0.158s (0.389s,  658.93/s)  Loss:  1.4648 (0.6512)  Acc@1:  65.625 ( 85.869)  Acc@5:  92.188 ( 97.399)                                                             
Test: [ 100/196]  Time: 0.778s (0.390s,  655.78/s)  Loss:  0.8628 (0.6884)  Acc@1:  78.516 ( 85.025)  Acc@5:  96.484 ( 97.084)                                                             
Test: [ 110/196]  Time: 0.158s (0.384s,  666.59/s)  Loss:  0.6924 (0.7007)  Acc@1:  87.109 ( 84.783)  Acc@5:  98.047 ( 96.970)                                                             
Test: [ 120/196]  Time: 0.158s (0.386s,  663.78/s)  Loss:  0.9434 (0.7065)  Acc@1:  80.859 ( 84.772)  Acc@5:  93.750 ( 96.843)                                                             
Test: [ 130/196]  Time: 0.158s (0.384s,  667.50/s)  Loss:  0.4998 (0.7252)  Acc@1:  89.062 ( 84.247)  Acc@5:  97.656 ( 96.642)                                                             
Test: [ 140/196]  Time: 0.159s (0.382s,  670.28/s)  Loss:  0.7427 (0.7355)  Acc@1:  84.766 ( 84.031)  Acc@5:  96.484 ( 96.562)                                                             
Test: [ 150/196]  Time: 0.159s (0.381s,  671.68/s)  Loss:  0.7056 (0.7445)  Acc@1:  86.719 ( 83.824)  Acc@5:  96.094 ( 96.456)                                                             
Test: [ 160/196]  Time: 0.159s (0.379s,  675.61/s)  Loss:  0.5107 (0.7516)  Acc@1:  89.844 ( 83.645)  Acc@5:  98.047 ( 96.378)                                                             
Test: [ 170/196]  Time: 0.158s (0.380s,  674.32/s)  Loss:  0.4509 (0.7618)  Acc@1:  91.016 ( 83.386)  Acc@5:  98.047 ( 96.281)                                                             
Test: [ 180/196]  Time: 0.159s (0.378s,  678.05/s)  Loss:  1.0498 (0.7705)  Acc@1:  76.562 ( 83.143)  Acc@5:  94.922 ( 96.228)                                                             
Test: [ 190/196]  Time: 0.159s (0.377s,  678.77/s)  Loss:  1.1494 (0.7721)  Acc@1:  74.219 ( 83.113)  Acc@5:  97.656 ( 96.249)                                                              
* Acc@1 83.166 (16.834) Acc@5 96.272 (3.728)                                                                 
--result                                                                        
{                                                                                    
"model": "map_convnext_tiny",                                                        
"top1": 83.166,                                                                 
"top1_err": 16.834,                                                                
"top5": 96.272,                                                                   
"top5_err": 3.728,                                                              
"param_count": 47.83,                                                             
"img_size": 224,                                                               
"cropt_pct": 0.875,                                                             
"interpolation": "bicubic"                                                             
}
```



</details>



<details>
    <summary>ConvNeXt-S</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py convnext_small -c 0,1 -m map_convnext_small
```



**Validation Command Line.** Running this command line will result in: *Acc@1 84.050 (15.950) Acc@5 96.668 (3.332)*

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_convnext_small --checkpoint ../output/result/map_convnext_small.pth.tar --crop-pct 0.875

Validating in mixed precision with native PyTorch AMP.
Loaded state_dict from checkpoint '../output/result/map_convnext_small.pth.tar'
Model map_convnext_small created, param count: 82837664
Data processing configuration for current model + dataset:                                                   input_size: (3, 224, 224)
        interpolation: bicubic
        mean: (0.485, 0.456, 0.406)
        std: (0.229, 0.224, 0.225)
        crop_pct: 0.875
        crop_mode: center
Test: [   0/196]  Time: 2.264s (2.264s,  113.07/s)  Loss:  0.4075 (0.4075)  Acc@1:  93.359 ( 93.359)  Acc@5:  98.438 ( 98.438)
Test: [  10/196]  Time: 0.230s (0.540s,  474.22/s)  Loss:  0.7920 (0.5697)  Acc@1:  82.812 ( 87.891)  Acc@5:  96.875 ( 98.082)
Test: [  20/196]  Time: 0.230s (0.467s,  548.75/s)  Loss:  0.4526 (0.5734)  Acc@1:  93.750 ( 88.114)  Acc@5:  97.266 ( 97.991)
Test: [  30/196]  Time: 0.231s (0.422s,  606.36/s)  Loss:  0.6523 (0.5364)  Acc@1:  88.672 ( 89.226)  Acc@5:  96.484 ( 98.059)
Test: [  40/196]  Time: 0.291s (0.411s,  622.15/s)  Loss:  0.4976 (0.5718)  Acc@1:  90.234 ( 88.310)  Acc@5:  98.047 ( 97.828)
Test: [  50/196]  Time: 0.231s (0.395s,  648.16/s)  Loss:  0.3381 (0.5730)  Acc@1:  94.922 ( 88.105)  Acc@5:  98.438 ( 97.871)
Test: [  60/196]  Time: 0.978s (0.408s,  627.74/s)  Loss:  0.7495 (0.5932)  Acc@1:  83.594 ( 87.699)  Acc@5:  97.266 ( 97.778)
Test: [  70/196]  Time: 0.231s (0.395s,  647.31/s)  Loss:  0.6494 (0.5802)  Acc@1:  85.156 ( 87.935)  Acc@5:  98.828 ( 97.926)
Test: [  80/196]  Time: 0.555s (0.397s,  644.14/s)  Loss:  0.9727 (0.5967)  Acc@1:  76.172 ( 87.543)  Acc@5:  95.703 ( 97.786)
Test: [  90/196]  Time: 0.231s (0.390s,  656.40/s)  Loss:  1.3926 (0.6252)  Acc@1:  68.750 ( 86.766)  Acc@5:  93.750 ( 97.558)
Test: [ 100/196]  Time: 0.685s (0.390s,  655.76/s)  Loss:  0.8296 (0.6594)  Acc@1:  80.859 ( 85.930)  Acc@5:  97.656 ( 97.331)
Test: [ 110/196]  Time: 0.231s (0.386s,  662.90/s)  Loss:  0.6191 (0.6713)  Acc@1:  86.719 ( 85.684)  Acc@5:  98.438 ( 97.213)
Test: [ 120/196]  Time: 0.231s (0.387s,  660.84/s)  Loss:  0.9072 (0.6742)  Acc@1:  80.078 ( 85.702)  Acc@5:  93.750 ( 97.111)
Test: [ 130/196]  Time: 0.231s (0.385s,  664.90/s)  Loss:  0.4678 (0.6933)  Acc@1:  89.844 ( 85.115)  Acc@5:  98.828 ( 96.979)
Test: [ 140/196]  Time: 0.231s (0.383s,  667.56/s)  Loss:  0.6221 (0.6996)  Acc@1:  86.719 ( 84.973)  Acc@5:  98.047 ( 96.914)
Test: [ 150/196]  Time: 0.231s (0.383s,  669.22/s)  Loss:  0.6587 (0.7083)  Acc@1:  86.719 ( 84.766)  Acc@5:  96.094 ( 96.828)
Test: [ 160/196]  Time: 0.233s (0.381s,  671.87/s)  Loss:  0.4668 (0.7163)  Acc@1:  90.234 ( 84.564)  Acc@5:  98.047 ( 96.749)
Test: [ 170/196]  Time: 0.232s (0.381s,  672.21/s)  Loss:  0.4465 (0.7258)  Acc@1:  91.016 ( 84.313)  Acc@5:  98.828 ( 96.660)
Test: [ 180/196]  Time: 0.232s (0.379s,  675.06/s)  Loss:  1.0576 (0.7355)  Acc@1:  73.438 ( 84.004)  Acc@5:  93.750 ( 96.625)
Test: [ 190/196]  Time: 0.232s (0.378s,  677.29/s)  Loss:  1.0264 (0.7365)  Acc@1:  76.562 ( 83.976)  Acc@5:  97.266 ( 96.632)
 * Acc@1 84.050 (15.950) Acc@5 96.668 (3.332)
--result
{
    "model": "map_convnext_small",
    "top1": 84.05,
    "top1_err": 15.95,
    "top5": 96.668,
    "top5_err": 3.332,
    "param_count": 82.84,
    "img_size": 224,
    "cropt_pct": 0.875,
    "interpolation": "bicubic"
}
```



</details>



<details>
    <summary>MaxViT-T</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py maxvit_tiny -c 0,1 -m map_maxvit_tiny_tf_224
```



**Validation Command Line.** Running this command line will result in: **

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_maxvit_tiny_tf_224 --checkpoint ../output/result/map_maxvit_tiny_tf_224.pth.tar --crop-pct 0.95

Validating in mixed precision with native PyTorch AMP.
Loaded state_dict from checkpoint '../output/result/map_maxvit_tiny_tf_224.pth.tar'
Model map_maxvit_tiny_tf_224 created, param count: 49958408
Data processing configuration for current model + dataset:
        input_size: (3, 224, 224)
        interpolation: bicubic
        mean: (0.485, 0.456, 0.406)
        std: (0.229, 0.224, 0.225)
        crop_pct: 0.95
        crop_mode: center
Test: [   0/196]  Time: 2.253s (2.253s,  113.63/s)  Loss:  0.8086 (0.8086)  Acc@1:  90.625 ( 90.625)  Acc@5:  98.047 ( 98.047)
Test: [  10/196]  Time: 0.287s (0.550s,  465.83/s)  Loss:  0.8350 (0.6176)  Acc@1:  80.469 ( 88.068)  Acc@5:  97.656 ( 98.295)
Test: [  20/196]  Time: 0.288s (0.458s,  559.24/s)  Loss:  0.4626 (0.5995)  Acc@1:  92.578 ( 88.467)  Acc@5:  97.266 ( 98.028)
Test: [  30/196]  Time: 0.287s (0.416s,  614.97/s)  Loss:  0.6895 (0.5658)  Acc@1:  87.891 ( 89.630)  Acc@5:  96.875 ( 98.122)
Test: [  40/196]  Time: 0.524s (0.406s,  631.04/s)  Loss:  0.5874 (0.6066)  Acc@1:  89.453 ( 88.586)  Acc@5:  97.656 ( 97.885)
Test: [  50/196]  Time: 0.289s (0.391s,  654.13/s)  Loss:  0.3916 (0.6071)  Acc@1:  94.531 ( 88.457)  Acc@5:  98.047 ( 97.924)
Test: [  60/196]  Time: 0.789s (0.401s,  638.02/s)  Loss:  0.7275 (0.6234)  Acc@1:  83.984 ( 88.006)  Acc@5:  96.094 ( 97.919)
Test: [  70/196]  Time: 0.288s (0.391s,  655.07/s)  Loss:  0.6660 (0.6093)  Acc@1:  87.109 ( 88.210)  Acc@5:  99.219 ( 98.041)
Test: [  80/196]  Time: 0.488s (0.391s,  654.53/s)  Loss:  1.0234 (0.6301)  Acc@1:  76.562 ( 87.780)  Acc@5:  95.703 ( 97.868)
Test: [  90/196]  Time: 0.289s (0.385s,  664.35/s)  Loss:  1.4121 (0.6555)  Acc@1:  65.234 ( 87.036)  Acc@5:  94.141 ( 97.678)
Test: [ 100/196]  Time: 0.626s (0.384s,  666.76/s)  Loss:  0.8306 (0.6882)  Acc@1:  78.906 ( 86.193)  Acc@5:  96.875 ( 97.428)
Test: [ 110/196]  Time: 0.290s (0.380s,  673.58/s)  Loss:  0.6528 (0.6982)  Acc@1:  86.719 ( 85.976)  Acc@5:  98.047 ( 97.347)
Test: [ 120/196]  Time: 0.384s (0.377s,  678.29/s)  Loss:  0.9229 (0.7006)  Acc@1:  80.859 ( 85.999)  Acc@5:  94.922 ( 97.266)
Test: [ 130/196]  Time: 0.290s (0.377s,  679.07/s)  Loss:  0.5000 (0.7192)  Acc@1:  90.625 ( 85.410)  Acc@5:  98.828 ( 97.137)
Test: [ 140/196]  Time: 0.290s (0.376s,  680.11/s)  Loss:  0.6929 (0.7266)  Acc@1:  86.328 ( 85.275)  Acc@5:  97.656 ( 97.094)
Test: [ 150/196]  Time: 0.290s (0.375s,  683.32/s)  Loss:  0.7080 (0.7342)  Acc@1:  87.109 ( 85.079)  Acc@5:  96.484 ( 97.038)
Test: [ 160/196]  Time: 0.290s (0.374s,  684.25/s)  Loss:  0.4592 (0.7419)  Acc@1:  92.188 ( 84.933)  Acc@5:  98.047 ( 96.941)
Test: [ 170/196]  Time: 0.291s (0.373s,  686.52/s)  Loss:  0.4602 (0.7520)  Acc@1:  92.188 ( 84.651)  Acc@5:  98.438 ( 96.889)
Test: [ 180/196]  Time: 0.292s (0.371s,  689.68/s)  Loss:  0.9453 (0.7631)  Acc@1:  78.906 ( 84.373)  Acc@5:  96.875 ( 96.847)
Test: [ 190/196]  Time: 0.291s (0.370s,  691.34/s)  Loss:  1.1465 (0.7688)  Acc@1:  74.219 ( 84.281)  Acc@5:  98.828 ( 96.853)
 * Acc@1 84.348 (15.652) Acc@5 96.876 (3.124)
--result
{
    "model": "map_maxvit_tiny_tf_224",
    "top1": 84.348,
    "top1_err": 15.652,
    "top5": 96.876,
    "top5_err": 3.124,
    "param_count": 49.96,
    "img_size": 224,
    "cropt_pct": 0.95,
    "interpolation": "bicubic"
}
```

​	

</details>



<details>
    <summary>FasterViT-3</summary>



**Train Command Line.**

```bash
CUDA_VISIBLE_DEVICES=0,1, torchrun --nproc_per_node=2 --master_port=12345 train_with_script.py faster_vit_3 -c 0,1 -m map_faster_vit_3_224
```



**Validation Command Line.** Running this command line will result in: *Acc@1 84.140 (15.860) Acc@5 96.652 (3.348)*

```bash
hankyul@hankyul:~$ CUDA_VISIBLE_DEVICES=0, python validate.py imageNet --model map_faster_vit_3_224 --checkpoint ../output/result/map_faster_vit_3_224.pth.tar --crop-pct 0.95

Validating in mixed precision with native PyTorch AMP.
/usr/local/lib/python3.10/dist-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Loaded state_dict from checkpoint '../output/result/map_faster_vit_3_224.pth.tar'
Model map_faster_vit_3_224 created, param count: 187338000
Data processing configuration for current model + dataset:
        input_size: (3, 224, 224)
        interpolation: bicubic
        mean: (0.485, 0.456, 0.406)                                                                         		std: (0.229, 0.224, 0.225)
        crop_pct: 0.95
        crop_mode: center
Test: [   0/196]  Time: 2.252s (2.252s,  113.66/s)  Loss:  0.5791 (0.5791)  Acc@1:  91.797 ( 91.797)  Acc@5:  98.047 ( 98.047)
Test: [  10/196]  Time: 0.287s (0.550s,  465.58/s)  Loss:  1.0273 (0.7488)  Acc@1:  80.859 ( 87.713)  Acc@5:  95.703 ( 97.834)
Test: [  20/196]  Time: 0.288s (0.463s,  553.34/s)  Loss:  0.6113 (0.7496)  Acc@1:  92.969 ( 87.723)  Acc@5:  96.875 ( 97.675)
Test: [  30/196]  Time: 0.289s (0.423s,  605.89/s)  Loss:  0.8291 (0.7121)  Acc@1:  87.500 ( 88.886)  Acc@5:  97.656 ( 97.921)
Test: [  40/196]  Time: 0.310s (0.405s,  632.57/s)  Loss:  0.7812 (0.7524)  Acc@1:  87.891 ( 87.910)  Acc@5:  97.656 ( 97.713)
Test: [  50/196]  Time: 0.287s (0.390s,  655.83/s)  Loss:  0.5962 (0.7538)  Acc@1:  94.531 ( 87.914)  Acc@5:  97.266 ( 97.756)
Test: [  60/196]  Time: 0.765s (0.401s,  639.12/s)  Loss:  0.9165 (0.7662)  Acc@1:  82.812 ( 87.487)  Acc@5:  96.094 ( 97.765)
Test: [  70/196]  Time: 0.289s (0.390s,  657.16/s)  Loss:  0.8433 (0.7511)  Acc@1:  86.328 ( 87.726)  Acc@5:  99.219 ( 97.942)
Test: [  80/196]  Time: 0.465s (0.389s,  657.38/s)  Loss:  1.2168 (0.7664)  Acc@1:  73.828 ( 87.375)  Acc@5:  95.312 ( 97.825)
Test: [  90/196]  Time: 0.289s (0.384s,  666.84/s)  Loss:  1.5352 (0.7934)  Acc@1:  67.969 ( 86.749)  Acc@5:  91.797 ( 97.536)
Test: [ 100/196]  Time: 0.631s (0.383s,  669.13/s)  Loss:  0.9883 (0.8241)  Acc@1:  79.297 ( 85.957)  Acc@5:  97.266 ( 97.297)
Test: [ 110/196]  Time: 0.290s (0.379s,  675.91/s)  Loss:  0.7856 (0.8344)  Acc@1:  89.062 ( 85.702)  Acc@5:  98.047 ( 97.213)
Test: [ 120/196]  Time: 0.290s (0.378s,  676.89/s)  Loss:  1.1211 (0.8372)  Acc@1:  78.516 ( 85.676)  Acc@5:  94.531 ( 97.130)
Test: [ 130/196]  Time: 0.289s (0.379s,  675.90/s)  Loss:  0.6089 (0.8547)  Acc@1:  90.234 ( 85.147)  Acc@5:  98.438 ( 96.994)
Test: [ 140/196]  Time: 0.290s (0.378s,  676.70/s)  Loss:  0.8213 (0.8596)  Acc@1:  86.328 ( 85.084)  Acc@5:  97.266 ( 96.933)
Test: [ 150/196]  Time: 0.292s (0.376s,  680.71/s)  Loss:  0.8613 (0.8699)  Acc@1:  86.328 ( 84.838)  Acc@5:  96.484 ( 96.826)
Test: [ 160/196]  Time: 0.291s (0.375s,  681.93/s)  Loss:  0.6108 (0.8773)  Acc@1:  90.625 ( 84.659)  Acc@5:  98.438 ( 96.749)
Test: [ 170/196]  Time: 0.292s (0.374s,  683.95/s)  Loss:  0.6099 (0.8883)  Acc@1:  92.188 ( 84.354)  Acc@5:  98.828 ( 96.679)
Test: [ 180/196]  Time: 0.291s (0.372s,  687.43/s)  Loss:  1.2012 (0.8974)  Acc@1:  78.906 ( 84.155)  Acc@5:  94.531 ( 96.614)
Test: [ 190/196]  Time: 0.291s (0.372s,  689.01/s)  Loss:  1.2158 (0.8989)  Acc@1:  73.438 ( 84.089)  Acc@5:  97.266 ( 96.619)
 * Acc@1 84.140 (15.860) Acc@5 96.652 (3.348)
--result
{
    "model": "map_faster_vit_3_224",
    "top1": 84.14,
    "top1_err": 15.86,
    "top5": 96.652,
    "top5_err": 3.348,
    "param_count": 187.34,
    "img_size": 224,
    "cropt_pct": 0.95,
    "interpolation": "bicubic"
}
```



</details>



## 4. Acknowledgement

This project is heavily based on [TIMM](https://github.com/huggingface/pytorch-image-models), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), [DeiT](https://github.com/facebookresearch/deit), and [PiT](https://github.com/naver-ai/pit). We sincerely appreciate the authors of the mentioned works for sharing such great library as open-source project.