import warnings
# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

from ml.SiamMAE import SiamMAE, CrossDecoder # Required to load the model

DEVICE = "cpu"
PATH = "../dataset_davis"
CLIP_DURATION = 20

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
    """
    Downloads and extracts the DAVIS 2017 dataset.

    This function checks if the DAVIS dataset directory already exists. If it doesn't, the function downloads the dataset from the official DAVIS website and saves it to the specified directory. After the dataset has been downloaded, the function extracts the zip file to the same directory and then removes the zip file.

    Parameters:
    dataset_path (str, optional): The directory where the dataset should be downloaded and extracted. Defaults to `"../dataset_davis"`.

    Usage:
    ```python
    download_davis(dataset_path="../dataset_davis")
    ```

    """
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

def unnormalize(tensor):
    """
    Unnormalizes a tensor by applying the reverse transformation of the normalization process.

    Args:
        tensor (torch.Tensor): The tensor to be unnormalized.

    Returns:
        torch.Tensor: The unnormalized tensor.

    """
    unnormalized = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return (torch.clamp(unnormalized, 0, 1) * 255).to(dtype=torch.uint8)

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
    """
    Creates a PyTorch DataLoader from a given dataset.

    This function takes a dataset and some optional parameters and returns a DataLoader that can be used to iterate over the dataset in batches.

    Parameters:
    data (Dataset): The dataset to be loaded.
    batch_size (int, optional): The number of samples per batch. Defaults to 1.
    num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.
    shuffle (bool, optional): Whether to shuffle the data after every epoch. Defaults to False.

    Returns:
    DataLoader: A PyTorch DataLoader for the given dataset.
    """
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

def compute_affinity(f1, f2, T = 1):
    """
    Computes the affinity between two feature maps.

    This function takes two feature maps, computes their matrix multiplication, and then applies a softmax function to the result. The temperature parameter T is used to control the sharpness of the softmax distribution.

    Parameters:
    f1 (torch.Tensor): The first feature map. Shape: [196, 768].
    f2 (torch.Tensor): The second feature map. Shape: [196, 768].
    T (float, optional): The temperature parameter for the softmax function. Defaults to 1.

    Returns:
    torch.Tensor: The computed affinity matrix. Shape: [786, 786].

    Usage:
    ```python
    affinity = compute_affinity(feature_map1, feature_map2, T=1)
    ```
    """
    # f1 = [196, 768]
    # f2 = [196, 768]
    # return aff = [786, 786]
    aff = torch.matmul(f1.T, f2)
    aff = torch.nn.functional.softmax(aff/T, dim=1)
    return aff # N x N

def create_masks(n, l, radius=5): # L x N x N == 196 x 768 x 768
    """
    Creates a tensor of masks for a given radius.

    This function generates a tensor of shape (l, n, n) where each slice along the first dimension is a mask that is True for all points within a specified radius of a certain point and False otherwise.

    Parameters:
    n (int): The height and width of each mask.
    l (int): The number of masks to generate.
    radius (int, optional): The radius within which points will be included in the mask. Defaults to 5.

    Returns:
    torch.Tensor: A tensor of shape (l, n, n) containing l masks.

    Usage:
    ```python
    masks = create_masks(768, 196, radius=5)
    ```
    """
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n))
    y = y.flatten()
    x = x.flatten()
    mask = torch.zeros((l, n, n))
    for i in range(l):
        mask[i] = (torch.sqrt((y - y[i])**2 + (x - x[i])**2) <= radius).reshape(n, n)
    return mask

def get_features(model, frame, upsample=True):
    """
    Extracts features from a given frame.

    Args:
        model (torch.nn.Module): The trained model.
        frame (torch.Tensor): The input frame.
        upsample (bool, optional): Whether to upsample the features. Defaults to True.

    Returns:
        torch.Tensor: The extracted features.
    """
    if frame.shape[0] == 3:
        features = frame.unsqueeze(0)
    else:
        features = frame
    
    with torch.no_grad():
        features = model.mae.encode(features)

    n = features.shape[1]
    img_size = int(np.sqrt(n))

    features = features.reshape(features.shape[0], img_size, img_size, -1).permute(0, 3, 1, 2)
    # scale to 224x224
    if upsample:
        features = nn.functional.interpolate(features, size=(frame.shape[2], frame.shape[3]), mode='nearest')
    return features

def create_mask(size, n):
    """
    Create a binary mask of size `size` where each pixel within a distance `n` from any other pixel is set to 1,
    and all other pixels are set to 0.

    Args:
        size (int): The size of the mask (width and height).
        n (float): The maximum distance between pixels for them to be considered neighbors.

    Returns:
        torch.Tensor: The binary mask of size `size`.
    """
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
    coords = torch.stack((x, y), dim=-1).reshape(-1, 2).float()

    dists = torch.cdist(coords, coords)
    mask = (dists <= n).float()

    return mask

def propagate_labels_multi_frames(features_previous, features_current, labels_previous, k, radius=5):
    """
    Propagates labels from previous frames to the current frame based on feature affinities.

    Args:
        features_previous (torch.Tensor): Feature maps of the previous frames. Shape: (n, 196, 768)
        features_current (torch.Tensor): Feature maps of the current frame. Shape: (196, 768)
        labels_previous (torch.Tensor): Labels of the previous frames. Shape: (n, 196, 3)
        k (int): Number of top affinities to consider for label propagation.
        radius (int, optional): Radius for creating the affinity mask. Defaults to 5.

    Returns:
        torch.Tensor: Propagated labels for the current frame. Shape: (196, 3)
    """
    T = 1 # temperature parameter

    # Reshape the feature maps
    features_previous = features_previous.reshape(features_previous.shape[0], features_previous.shape[1], -1).transpose(1, 2) # n x 196 x 768
    features_current = features_current.reshape(features_current.shape[0], -1).transpose(0, 1)                                # 196 x 768
    labels_previous = labels_previous.reshape(labels_previous.shape[0], labels_previous.shape[1], -1).transpose(1, 2)         # n x 196 x 3

    affinities = []
    for frame_nr in range(features_previous.shape[0]):
        # Compute the affinity between the features in the previous and current frames
        affinity = torch.matmul(features_previous[frame_nr, :, :], features_current.T) # 196 x 196
        affinity = torch.nn.functional.softmax(affinity/T, dim=1)
        mask = create_mask(int(affinity.shape[0]**0.5), radius)

        affinity = affinity * mask

        affinities.append(affinity)
    
    affinities = torch.stack(affinities, dim=0) # shape: n x 196 x 196

    labels_next = torch.zeros((labels_previous.shape[1], labels_previous.shape[2]))
    for i in range(labels_previous.shape[1]): # loop over all pixels
        averaged_value = torch.zeros((labels_previous.shape[2]))
        total_weight = 0
        for j in range(labels_previous.shape[0]):
            value, index = torch.sort(affinities[j, :, i])
            
            value = value[-k:]
            index = index[-k:]

            averaged_value += torch.matmul(value, labels_previous[j, index, :])
            total_weight += torch.sum(value)
        averaged_value /= total_weight
        labels_next[i, :] = averaged_value

    return labels_next


def evaluate_video(model, video, annotation, queue_length=20, k=7, neighbor=1, interpolation_mode='bilinear'):
    """
    Evaluate the video by calculating the Jaccard index and F1 score for each frame. Labels are propagated from
    the previous frames to the current frame.

    Args:
        model (torch.nn.Module): The trained model.
        video (torch.Tensor): The video frames.
        annotation (torch.Tensor): The ground truth annotation for each frame.
        queue_length (int, optional): The length of the queue frames. Defaults to 20.
        k (int, optional): The number of nearest neighbors to consider. Defaults to 7.
        neighbor (int, optional): The number of neighbors to propagate labels from. Defaults to 1.
        interpolation_mode (str, optional): The interpolation mode for resizing frames. Defaults to 'bilinear'.

    Returns:
        float: The average Jaccard index for all frames.
        float: The average F1 score for all frames.
    """
    video_length = video.shape[0]
    
    # Calculate features for all frames
    features = get_features(model, video, upsample=False)
    
    jaccard = torchmetrics.classification.MultilabelJaccardIndex(3, average='micro', validate_args=False)
    f1 = torchmetrics.classification.MultilabelF1Score(3, average='micro')
    for i in range(video_length-queue_length):
        # Prepare queue frames
        features_previous = features[i:i+queue_length]
        labels_previous = nn.functional.interpolate(torch.stack([annotation[i+j] for j in range(queue_length)]).unsqueeze(0), size=(3, 14, 14), mode='nearest').squeeze(0)

        # Prepare next frame
        features_next = features[i+queue_length]

        # Propagate labels
        labels_next = propagate_labels_multi_frames(features_previous, features_next, labels_previous, k, neighbor).reshape(14, 14, 3).permute(2, 0, 1)

        # Calculate jaccard index
        next_labels = nn.functional.interpolate(unnormalize(labels_next).unsqueeze(0), size=(224, 224), mode=interpolation_mode).squeeze(0).permute(1, 2, 0)

        ground_truth = unnormalize(annotation[i+queue_length]).permute(1, 2, 0)
        ground_truth = ((ground_truth / ground_truth.max()) > 0.5).to(dtype=torch.uint8)
        
        prediction = ((next_labels/next_labels.max()) > 0.5).to(dtype=torch.uint8)
        # Choose the class with the highest probability (one-hot encoding)
        #prediction = torch.zeros_like(next_labels)
        #prediction[torch.arange(next_labels.shape[0]), torch.arange(next_labels.shape[1]), next_labels.argmax(dim=2)] = 1

        val = jaccard(prediction.permute(2, 0, 1).unsqueeze(0), ground_truth.permute(2, 0, 1).unsqueeze(0))
        f1(prediction.permute(2, 0, 1).unsqueeze(0), ground_truth.permute(2, 0, 1).unsqueeze(0))
        #plt.subplot(1, 2, 1)
        #plt.imshow(prediction*255)
        #plt.subplot(1, 2, 2)
        #plt.imshow(ground_truth*255)
        #plt.title(f"Jaccard Index: {val.item():.4f}")
        #plt.show()

    return jaccard.compute().item(), f1.compute().item()

def description(model_name, jaccard, f1):
    return f"{model_name} | Mean Jaccard Index: {np.mean(jaccard):.4f} Mean F1 Score: {np.mean(f1):.4f}"

def run_experiment_for_model(dloader, annot_loader, num_videos, checkpoint_path, model_name, results_file="../dataset_davis/results.txt"):
    """
    Runs an experiment for a given model and writes the results to a file.

    This function loads a model from a checkpoint, iterates over a dataset and its corresponding annotations, evaluates the model on each video in the dataset, and writes the results to a file.

    Parameters:
    dloader (DataLoader): The DataLoader for the video dataset.
    annot_loader (DataLoader): The DataLoader for the video annotations.
    num_videos (int): The number of videos in the dataset.
    checkpoint_path (str): The path to the model checkpoint.
    model_name (str): The name of the model.
    results_file (str, optional): The file to write the results to. Defaults to "../dataset_davis/results.txt".
    """
    model = torch.load(checkpoint_path).to(DEVICE)
    
    pbar = tqdm(zip(dloader, annot_loader), total=num_videos, desc=description(model_name, [0.0], [0.0]))
    js = []
    f1s = []
    for batch, annot_batch in pbar:
        video = batch["video"]
        annotation = annot_batch["video"]

        jaccard, f1 = evaluate_video(model, video[0].to(DEVICE), annotation[0].to(DEVICE), queue_length=20, k=7, neighbor=1, interpolation_mode='bilinear')
        js.append(jaccard)
        f1s.append(f1)

        pbar.set_description(description(model_name, js, f1s))

    print(description(model_name, js, f1s))
    
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "a") as f:
        f.write(description(model_name, js, f1s) + "\n")

    del model


if __name__ == "__main__":
    model_names = ["5k pretrained base 75%", "5k pretrained base 95%","5k pretrained base 50%"]
    checkpoints = ["../checkpoints/5k_2023-11-27_12:55:32/10.pt","../checkpoints/5k_95_2023-12-04_15:13:38/10.pt", "../checkpoints/5k_50_2023-11-27_19:05:02/10.pt"]



    label = {"category": "example"}
    data_list = create_label_list(f"{PATH}/DAVIS/JPEGImages/480p/", label)
    annotation_list = create_label_list(f"{PATH}/DAVIS/Annotations/480p/", label)

    data = LabeledVideoDataset(data_list, clip_sampler=UniformClipSampler(clip_duration=CLIP_DURATION), decode_audio=False, transform=TRANSFORM)
    annotations = LabeledVideoDataset(annotation_list, clip_sampler=UniformClipSampler(clip_duration=CLIP_DURATION), decode_audio=False, transform=TRANSFORM)
    
    num_videos = data.num_videos

    
    for checkpoint, model_name in zip(checkpoints, model_names):
        dloader = create_dataloader(data)
        annot_loader = create_dataloader(annotations)
        run_experiment_for_model(dloader, annot_loader, num_videos, checkpoint, model_name)