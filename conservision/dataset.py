import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame | None = None, transforms=None):
        self.data = x_df
        self.label = y_df
        if transforms is not None:
            self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        image_id = self.data.index[index]
        sample = {"image_id": image_id, "image": image}

        if self.label is not None:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample |= {"label": label}

        return sample

    def __len__(self):
        return len(self.data)