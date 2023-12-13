import os
import requests
import zipfile
import torch
import torch.nn as nn
import numpy as np
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    Div255,
    Permute
)

from torchvision.transforms import (
    Compose,
    CenterCrop,
)
import torchmetrics
from tqdm import tqdm

import sys
sys.path.append("..") # Root path of the project

from ml.SiamMAE import SiamMAE, CrossDecoder


TRANSFORM = Compose(
    [
    ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
            CenterCrop((224, 224)),
            Div255(),
            # mean and std from ImageNet
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # Permute to get Frames, Channel, Height, Width
            Permute((1, 0, 2, 3))
            ]
        ),
        ),
    ]
)

def download_davis(dataset_path="../dataset_davis"):
    path_to_zip = os.path.join(dataset_path, "DAVIS-2017-trainval-480p.zip")

    if not os.path.exists(dataset_path):
        content = requests.get("https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip")
        os.makedirs(dataset_path, exist_ok=True)
        with open(path_to_zip, "wb") as f:
            f.write(content.content)
        print(f"Successfully downloaded DAVIS dataset to {path_to_zip}")

        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
            
        print(f"Unzipped DAVIS dataset to {dataset_path}")
        os.remove(path_to_zip)
        print("Finished downloading and extracting DAVIS dataset")

def create_label_list(directory_path, label):
    """
    Create a list of labeled directories (necessary for loading videos as directories
    of frames).

    Args:
        directory_path (str): The path to the directory containing the directories to be labeled.
        label (dict): A dictionary containing the label to be applied to the directories.

    Returns:
        list: A list of tuples, where each tuple contains the path of a labeled directory and its corresponding label.
    """
    items = os.listdir(directory_path)
    dirs = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    labeled_dirs = [(os.path.join(directory_path, dir), label) for dir in dirs]

    return labeled_dirs
    
def create_dataloader(data, batch_size=1, num_workers=0, shuffle=False):
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

def evaluate_video(video, annotation, pos_embed_factor=1.0, queue_length=20, k=7, neighbor=1, interpolation_mode='bilinear'):
    jaccard = torchmetrics.classification.MultilabelJaccardIndex(3, average='micro', validate_args=False)
    f1 = torchmetrics.classification.MultilabelF1Score(3, average='micro')

    return (0.0, 0.0)

if __name__ == "__main__":
    PATH = "../dataset_davis"
    MODEL_NAME = "5k pretrained base 75%"
    CHECKPOINT_PATH = "../checkpoints/5k_2023-11-27_12:55:32/10.pt"
    CLIP_DURATION = 20

    label = {"category": "example"}
    data_list = create_label_list(f"{PATH}/DAVIS/JPEGImages/480p/", label)
    annotation_list = create_label_list(f"{PATH}/DAVIS/Annotations/480p/", label)

    data = LabeledVideoDataset(data_list, clip_sampler=UniformClipSampler(clip_duration=CLIP_DURATION), decode_audio=False, transform=TRANSFORM)
    annotations = LabeledVideoDataset(annotation_list, clip_sampler=UniformClipSampler(clip_duration=CLIP_DURATION), decode_audio=False, transform=TRANSFORM)

    dloader = create_dataloader(data)
    annot_loader = create_dataloader(annotations)
    
    model = torch.load(CHECKPOINT_PATH)
    
    pbar = tqdm(zip(dloader, annot_loader), total=data.num_videos)
    js = []
    f1s = []
    for batch, annot_batch in pbar:
        video = batch["video"]
        annotation = annot_batch["video"]

        jaccard, f1 = evaluate_video(video[0], annotation[0], pos_embed_factor=1.0, interpolation_mode='bilinear')
        js.append(jaccard)
        f1s.append(f1)

        pbar.set_description(f"Mean Jaccard Index: {np.mean(js):.4f} Mean F1 Score: {np.mean(f1s):.4f}")
        break
    
    del model

    print(f"Mean Jaccard Index: {np.mean(js):.4f} | Mean F1 Score: {np.mean(f1s):.4f}")