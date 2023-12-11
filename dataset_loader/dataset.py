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

from .random_temporal_subsample import RandomTemporalSubsample

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip
)

# Constants for video processing
FRAME_GAP_RANGE = (4, 48)  # Range of possible gaps between frames
FPS = 29  # Frames per second in the video
IMAGE_SIZE = 224  # Size of the video frames after resizing
REPEATED_SAMPLING_FACTOR = 1  # Number of times to repeat the sampling process
CLIP_DURATION = FRAME_GAP_RANGE[1] / FPS + 0.0001  # Duration of each video clip

def get_kinetics_dataset(dataset_dir="../dataset/"):
    """
    Loads the Kinetics dataset from a specified directory and applies a series of transformations to the video data.

    Parameters:
    dataset_dir (str): The directory where the Kinetics dataset is stored.

    Returns:
    Kinetics: The transformed Kinetics dataset.
    """
    # Print a message indicating where the dataset is being loaded from
    print(f"Loading dataset from {dataset_dir}")

    # Define the transformations to be applied to the video data
    transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # Randomly subsample the video along the temporal dimension
                    RandomTemporalSubsample(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1], repeated_sampling=REPEATED_SAMPLING_FACTOR),
                    # Resize and crop the video frames
                    RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.2, 1.0), aspect_ratio=(1.0, 1.0), interpolation='bilinear'),
                    # Divide the pixel values by 255 to normalize them to the range [0, 1]
                    Div255(),
                    # Normalize the pixel values using the mean and standard deviation from ImageNet
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    # Permute the dimensions of the video tensor to get [Frames X Channel X Height X Width]
                    Permute((1, 0, 2, 3)),
                    # Randomly flip the video frames horizontally
                    RandomHorizontalFlip(p=0.5)
                ]
            ),
        ),
    ])

    # Return the Kinetics dataset with the transformations applied
    return Kinetics(dataset_dir, clip_sampler=RandomClipSampler(clip_duration=CLIP_DURATION), decode_audio=False, transform=transform)

def unnormalize(tensor):
    """
    Unnormalizes a tensor that was previously normalized with the mean and standard deviation from ImageNet.

    Parameters:
    tensor (torch.Tensor): The tensor to unnormalize.

    Returns:
    torch.Tensor: The unnormalized tensor.
    """
    return tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

def unnormalize_for_plot(tensor):
    """
    Unnormalizes a tensor that was previously normalized with the mean and standard deviation from ImageNet, 
    and prepares it for plotting by detaching it, moving it to the CPU, and permuting its dimensions.

    Parameters:
    tensor (torch.Tensor): The tensor to unnormalize and prepare for plotting.

    Returns:
    numpy.ndarray: The unnormalized tensor as a NumPy array, ready for plotting.
    """
    return unnormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy()
