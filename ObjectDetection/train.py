"""
Main file for training Yolo model on Pascal VOC datasets
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import VOCDataset
from model import Yolov1
from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
)
from utils import YoloLoss
from pathlib import Path

seed = 123
torch.manual_seed(seed)

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
# DEVICE = "cpu"

# Hyperparameters etc.
BASE_DIR = Path(__file__).parent.parent

LEARNING_RATE = 2e-5
BATCH_SIZE = 16  # 64
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False  # load model from DATA_DIR

LOAD_MODEL_FILE = BASE_DIR.joinpath("overfit.pth.tar")  # save overfitted
DATA_DIR = BASE_DIR.joinpath("data")
IMG_DIR = DATA_DIR.joinpath("images")
LABEL_DIR = DATA_DIR.joinpath("labels")


class Compose(object):
    def __init__(self, my_transforms):
        self.transforms = my_transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])  # reshape img


def train(train_loader: DataLoader, model: nn.Module, optimizer: torch.optim, loss_fn: nn.Module):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")
