"""
Code for calculating the mean and standard deviation of a dataset.
This is useful for normalizing the dataset to obtain mean 0, std 1. 

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-05-09 Initial coding
*    2022-12-16 Updated comments, code revision, and checked code still works with latest PyTorch.

"""
import sys
from pathlib import Path
import fire

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path.cwd().parents[0]))
from conservision.dataset import ImagesDataset
from utils.transforms import BASIC_TRANSFORM


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std

def main(
        train_features_csv: str,
        test_features_csv: str,
        train_images_root: str,
        test_images_root: str,
        batch_size: int = 256,
    ) -> None:

    features = pd.read_csv(train_features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: Path(train_images_root) / str(path)) # type: ignore

    x = features.filepath.to_frame()

    dataset = ImagesDataset(x, transforms=BASIC_TRANSFORM)

    dataloader = DataLoader(dataset, batch_size=batch_size)


    mean, std = get_mean_std(dataloader)
    print(mean)
    print(std)