# https://s3.amazonaws.com/kinetics/400/train/part_0.tar.gz
# https://s3.amazonaws.com/kinetics/400/train/part_{i}.tar.gz
# max i is 242

import os
import tarfile 
import subprocess

from torchvision.datasets.utils import download_url
from tqdm import tqdm
import cv2

MAX_EXTRACTS = 242
MIN_SECONDS_PER_VIDEO = 3
DEFULT_DATASET_DIR = "../dataset"
TMP_FILE_DOWNLOADED_PARTS = f"{DEFULT_DATASET_DIR}/parts.txt"

# DOWNLOAD THE DATASET FROM THE URL AND EXTRACT IT TO THE SPECIFIED DIRECTORY
def download_kinetics(dataset_dir=DEFULT_DATASET_DIR, max_extractions=10): 
    """
    Downloads and extracts the Kinetics dataset from the internet. 
    The function downloads the dataset in parts and extracts them into the 'class1' directory. 
    It keeps track of the downloaded parts in a temporary file to avoid re-downloading.

    Args:
        dataset_dir (str, optional): The directory where the Kinetics dataset will be stored. Defaults to DEFULT_DATASET_DIR.
        max_extractions (int, optional): The maximum number of parts to download and extract. Defaults to 1.

    Returns:
        None. The function modifies the filesystem by downloading and extracting files.
    """        
    # Download the dataset from the internet
    downloaded_parts = []
    if os.path.exists(TMP_FILE_DOWNLOADED_PARTS):
        with open(TMP_FILE_DOWNLOADED_PARTS, "r") as f:
            downloaded_parts = f.read().split(",")
        
    # We put in class1 because we don't care about the class we just want the videos
    dataset_dir = dataset_dir + "/class1"

    max_extractions = min(max_extractions, MAX_EXTRACTS)
    for i in tqdm(range(max_extractions)):
        # Check if the part is already downloaded
        if str(i) in downloaded_parts:
            continue

        download_url(f"https://s3.amazonaws.com/kinetics/400/train/part_{i}.tar.gz", root=dataset_dir)  
        # Extract the dataset
        with tarfile.open(f"{dataset_dir}/part_{i}.tar.gz", "r:gz") as tar:
            tar.extractall(path=dataset_dir)
        # Delete the tar.gz file
        os.remove(f"{dataset_dir}/part_{i}.tar.gz")

    # Create a text file with the last part extracted
    with open(TMP_FILE_DOWNLOADED_PARTS, "w+") as f:
        f.write(",".join([str(i) for i in range(max_extractions)]))
    return


# Function to get fps and duration of a video file
def _get_video_info(file_path):
    # Use cv2 to find fps and duration of video
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return fps, duration

def validate_kinetics(dataset_dir=DEFULT_DATASET_DIR):
    """
    Validates the Kinetics dataset by checking if it's downloaded and extracted correctly.
    It also checks each video file in the 'class1' directory for corruption, duration, and FPS range.
    Corrupted files, files with duration less than MIN_SECONDS_PER_VIDEO, and files with FPS not in the range of MIN_FPS to MAX_FPS are deleted.

    Args:
        dataset_dir (str, optional): The directory where the Kinetics dataset is stored. Defaults to DEFULT_DATASET_DIR.

    Returns:
        None. Prints messages for each validation step and for any issues found.
    """   
    # Check if the dataset is downloaded
    if not os.path.exists(dataset_dir):
        return print("Dataset not downloaded")
          
    # Check if the dataset is extracted
    if not os.path.exists(dataset_dir + "/class1"):
        return print("Dataset not extracted")

    for file in tqdm(os.listdir(dataset_dir + "/class1")):
        # Check if the file is a video
        if file.endswith(".mp4"):
            # Check if the file is corrupted
            try:
                fps, duration = _get_video_info(dataset_dir + "/class1/" + file)
            except:
                print(f"File {file} is corrupted")
                # Delete the corrupted file
                os.remove(dataset_dir + "/class1/" + file)
                continue
            
            # Remove videos that are less than MIN_SECONDS_PER_VIDEO seconds
            if duration < MIN_SECONDS_PER_VIDEO:
                os.remove(dataset_dir + "/class1/" + file)
                continue

            if not (fps >= 29 and fps <= 30) or (fps < 23.9 and fps > 29):
                os.remove(dataset_dir + "/class1/" + file)
                continue

if __name__ =="__main__":
    print("Downloading Kinetics Dataset")
    download_kinetics()
    
    print("Validating Kinetics Dataset")
    validate_kinetics()

    print("Done!")