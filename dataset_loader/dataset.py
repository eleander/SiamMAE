import torch
from torch.utils.data import Dataset
import cv2
import os

from pytorchvideo.data import Kinetics, RandomClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomResizedCrop,
    Div255,
    Permute
)

from .random_temporal_subsample import RandomTemporalSubsample, LinearTemporalSubsample

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip
)

FRAME_GAP_RANGE = (2, 10)
FPS = 29
IMAGE_SIZE = 224
REPEATED_SAMPLING_FACTOR = 3
CLIP_DURATION = FRAME_GAP_RANGE[1] / FPS + 0.0001

def get_kinetics_dataset(dataset_dir="../dataset/"):
    print(f"Loading dataset from {dataset_dir}")
    transform = Compose(
    [
    ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
            # RandomTemporalSubsample(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1], repeated_sampling=REPEATED_SAMPLING_FACTOR),
            LinearTemporalSubsample(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1], repeated_sampling=REPEATED_SAMPLING_FACTOR),
            RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.2, 1.0), aspect_ratio=(1.0, 1.0), interpolation='bilinear'),
            Div255(),
            # mean and std from ImageNet
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # Permute to get [Frames X Channel X Height X Width]
            Permute((1, 0, 2, 3)),
            RandomHorizontalFlip(p=0.5)
            ]
        ),
        ),
    ]   
    )
    return Kinetics(dataset_dir, clip_sampler=RandomClipSampler(clip_duration=CLIP_DURATION), decode_audio=False, transform=transform)

def unnormalize(tensor):
    return tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

def unnormalize_for_plot(tensor):
    return unnormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy()
