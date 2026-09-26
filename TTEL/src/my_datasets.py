from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from itertools import product
from pathlib import Path


class MNIST(Dataset):
    def __init__(self, mode="train", path="/home/shared/data/MNIST1"):
        self.data = torch.load(path + f"/{mode}.pt")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        return self.data["images"][idx].permute(1, 2, 0), self.data["labels"][idx]


class COCO(Dataset):
    def __init__(self, data_path="../data", mode="train"):

        data = torch.load(f"{data_path}/coco/coco_dino_{mode}.pt", map_location="cpu")
        self.embeddings = data["embeddings"]
        self.path = data["paths"]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.path[index]


class PreSavedBatchDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, indices):
        self.embeddings = embeddings[indices]
        self.labels = labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class VisualGenome(Dataset):

    def __init__(
        self,
        split="id",
        data_dir="~/genome",
    ):
        DATA_DIR = Path.home() / "genome"

        self.data_dir = Path(data_dir).expanduser()
        self.split = split

        data = torch.load(
            DATA_DIR / f"{split}_representations.pt",
            weights_only=False,
        )

        self.image_ids = data["image_ids"]
        self.representations = data["representations"]

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return self.representations[idx], self.image_ids[idx]
