import sys
from pathlib import Path
from typing import Any, Iterable

import fire
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

sys.path.append(str(Path.cwd().parents[0]))
from conservision.dataset import ImagesDataset
from conservision.models import resnet50_animal
from utils.load import load_training_data
from utils.transforms import BASIC_TRANSFORM
from utils.augmentations import AUGMENTATIONS


def train_step(model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader, augmentations: Any, device: torch.device):
    model = model.train()
    for batch in tqdm(dataloader, desc="training"):
        x = batch['image'].to(device)
        y = batch['label'].to(device)

        a = model(x)
        loss = criterion(a, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_step(model: nn.Module, criterion: nn.CrossEntropyLoss, dataloader: DataLoader, device: torch.device):
    model = model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="evaluation"):
        x = batch['image'].to(device)
        y = batch['label'].to(device)
        
        with torch.no_grad():
            a = model(x)
            loss = criterion(a, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(model: nn.Module, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, train_dataloader: DataLoader, eval_dataloader: DataLoader, num_epochs: int, device: torch.device):
    model = model.to(device)
    for epoch in range(1, num_epochs+1):
        train_step(model, criterion, optimizer, train_dataloader, AUGMENTATIONS, device)
        eval_loss = eval_step(model, criterion, eval_dataloader, device)
        print(f'{eval_loss = }')
    lr_scheduler.step()


def main(features_path: str, labels_path: str, images_root: str, config_path: str, model_save_path: str):
    # Hyperparams
    batch_size = 32

    device = torch.device('cuda')

    # File setups:
    Path(model_save_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Load Dataset
    x, y = load_training_data(features_csv=features_path, labels_csv=labels_path, images_root=images_root)
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    train_dataset = ImagesDataset(x_train, y_train, BASIC_TRANSFORM)
    eval_dataset = ImagesDataset(x_eval, y_eval, BASIC_TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=2)
    model = resnet50_animal()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    train(nn.DataParallel(model), criterion, optimizer, lr_scheduler, train_dataloader, eval_dataloader, num_epochs=1, device=device)
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    fire.Fire(main)