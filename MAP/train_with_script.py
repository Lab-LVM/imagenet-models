import argparse
import gc
import os

import torch
import wandb
from timm.utils import setup_default_logging

from train import _parse_args, run
from models import *

setting_dict = dict(
    resnet50="imageNet --input-size 3 224 224 --test-input-size 3 224 224 --aa rand-m20-mstd0.5-inc1 --mixup .1 --cutmix 1.0 --remode pixel --reprob 0.25 --crop-pct 0.95 --drop-path 0.1 --drop 0.1 --smoothing 0.1 --bce-loss --opt lamb --weight-decay .02 --sched cosine --epochs 300 --lr 5e-3 --warmup-lr 1e-6 -b 128 -j 8 --channels-last --amp -tb 1024 --pin-mem --aug-repeats 3 --log-wandb",
    pit_s="imageNet --model vit_small_patch16_224 --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 3 --remode pixel --reprob 0.25 --drop-path .1 --opt adamw --weight-decay .05 --sched cosine --epochs 300 --lr 1e-3 --warmup-lr 1e-6 -b 256 -tb 1024 -j 16 --amp --channels-last --log-wandb --pin-mem",
    convnext_tiny="imageNet --drop-path .1 -b 128 -tb 1024 --smoothing 0.1 --bce-loss --opt lamb --opt-eps 1e-8 --momentum 0.8 --weight-decay 0.05 --sched cosine --epochs 300 --lr 5e-3 --warmup-lr 1e-6 --crop-pct 0.875 --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --remode pixel --reprob 0.25 --sched cosine -j 8 --amp --channels-last --model-ema --model-ema-decay 0.9999 --aug-repeats 3 --log-wandb",
    convnext_small="imageNet --drop-path .4 -b 128 -tb 1024 --smoothing 0.1 --bce-loss --opt lamb --opt-eps 1e-8 --momentum 0.8 --weight-decay 0.05 --sched cosine --epochs 300 --lr 5e-3 --warmup-lr 1e-6 --crop-pct 0.875 --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --remode pixel --reprob 0.25 --sched cosine -j 8 --amp --channels-last --model-ema --model-ema-decay 0.9999 --aug-repeats 3 --log-wandb",
    faster_vit_3="imageNet --drop-path .3 -b 128 -tb 4096 --aug-repeat 3 --opt lamb --opt-eps 1e-8 --momentum 0.9 --weight-decay 0.05 --sched cosine --warmup-epochs 35 --epochs 300 --lr 5e-3 --warmup-lr 1e-6 --min-lr 5e-6 --crop-pct 0.95 --aa rand-m15-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --remode pixel --reprob 0.25 --smoothing 0.1 --sched cosine -j 8 --amp --channels-last --log-wandb --clip-grad 5.0",
    maxvit_tiny="imageNet --model maxvit_tiny_tf_224 --aug-repeat 3 --aa rand-m15-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --remode pixel --reprob 0.25 --drop-path .2 --opt lamb --bce-loss --weight-decay .05 --sched cosine --epochs 300 --lr 8e-3 --warmup-lr 1e-6 --warmup-epoch 30 --min-lr 1e-5 -b 64 -tb 4096 --smoothing 0.1 --clip-grad 1.0 -j 8 --amp --pin-mem --channels-last --log-wandb --project-name mmcap",
    mobilenet_v1="imageNet --input-size 3 160 160 --test-input-size 3 224 224 --aa rand-m7-mstd0.5-inc1 --mixup .1 --cutmix 1.0 --aug-repeats 0 --remode pixel --reprob 0.0 --crop-pct 0.95 --drop-path 0.05 --smoothing 0.0 --bce-loss --opt lamb --weight-decay .02 --sched cosine --epochs 100 --lr 5e-3 --warmup-lr 1e-6 -b 512 -j 16 --channels-last --amp -tb 1024 --pin-mem --log-wandb",
)


def get_multi_args_parser():
    parser = argparse.ArgumentParser(description='timm-multi-run', add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('setup', type=str, nargs='+', choices=setting_dict.keys(), help='experiment setup')
    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['resnet50'], help='list of model names')
    parser.add_argument('-c', '--cuda', type=str, default='0,', help='gpu device ids')
    parser.add_argument('-r', '--resume', type=str, default=None, help='resume checkpoint path')
    parser.add_argument('-ri', '--resume_id', type=str, default=None, help='resume id')
    parser.add_argument('-cp', '--initial-checkpoint', type=str, default=None, help='initial checkpoint path')
    parser.add_argument('-fp', '--finetuning-checkpoint', type=str, default=None, help='finetuning checkpoint path')
    parser.add_argument('-pt', '--in21k-to-in1k', action='store_true', help='convert in21k->in1k fc weight and bias')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('-e', '--eval-epoch', type=int, default=1, help='evaluate model after this epoch')
    parser.add_argument('-es', '--early-stop', type=int, default=None, help='early stop model training')

    # mmcap hyper-parameter (fixed)
    parser.add_argument('--dec-lam', default=-0.8, type=float)
    parser.add_argument('--distill-tokens', default=0, type=float)
    parser.add_argument('--token-distillation', default=1, type=float)

    return parser


def clear(is_master):
    # 1. clear gpu memory
    torch.cuda.empty_cache()
    # 2. clear cpu memory
    gc.collect()
    # 3. close logger
    if is_master:
        wandb.finish(quiet=True)


if __name__ == '__main__':
    setup_default_logging()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_master = local_rank == 0
    multi_args_parser = get_multi_args_parser()
    multi_args = multi_args_parser.parse_args()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = multi_args.cuda
    # os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

    for setup in multi_args.setup:
        args, args_text = _parse_args(setting_dict[setup].split())
        for model in multi_args.model_name:
            args.local_rank = local_rank
            args.model = model
            args.resume = multi_args.resume
            args.resume_id = multi_args.resume_id
            args.initial_checkpoint = multi_args.initial_checkpoint
            args.finetuning_checkpoint = multi_args.finetuning_checkpoint
            args.in21k_to_in1k = multi_args.in21k_to_in1k
            args.seed = multi_args.seed
            args.eval_epoch = multi_args.eval_epoch
            args.early_stop = multi_args.early_stop
            args.dec_lam = multi_args.dec_lam
            args.distill_tokens = multi_args.distill_tokens
            args.token_distillation = multi_args.token_distillation
            run(args, args_text)
            clear(is_master)
