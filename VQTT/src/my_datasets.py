from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import torch
import shapeworld
from PIL import Image
import math
import os
import pickle
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from itertools import product
from dotenv import load_dotenv



class ShapeWorld(Dataset):
    def __init__(
        self,
        n=128,
        mode="train",
        dtype="agreement",
        name="existential",
        collision_tolerance=0.0,
        config="/home/shared/ShapeWorld/configs/agreement/existential/oneshape_test.json",
        **kwargs,
    ):
        dataset = shapeworld.Dataset.create(
            dtype=dtype,
            name=name,
            collision_tolerance=collision_tolerance,
            config=config,
            **kwargs,
        )
        generated = dataset.generate(n=n, mode=mode, include_model=True)
        self.imgs = generated["world"]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.imgs[idx]), torch.tensor(1).to(device="cuda")


class MNIST(Dataset):
    def __init__(self, mode="train", path="/home/shared/data/MNIST1"):
        self.data = torch.load(path + f"/{mode}.pt")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        return self.data["images"][idx].permute(1, 2, 0), self.data["labels"][idx]


class COCO(Dataset):
    def __init__(self, data_path="../data", mode="train"):

        data = torch.load(
            f"{data_path}/coco/coco_dino_{mode}.pt", map_location="cpu"
        )
        self.embeddings = data["embeddings"]
        self.path = data["paths"]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.path[index]


def generate_shape(n=10000, train_split=0.8, val_split=0.1, test_split=0.1):
    def save_dataset(dataset, path):
        os.makedirs(path, exist_ok=True)
        data = []
        for example in dataset:
            data.append(example)
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)

    train_dataset = ShapeWorld(n=math.floor(n * train_split), mode="train")
    val_dataset = ShapeWorld(n=math.floor(n * val_split), mode="validation")
    test_dataset = ShapeWorld(n=math.floor(n * test_split), mode="test")
    
    load_dotenv()
    DATA_PATH = os.getenv("DATA_PATH")

    save_dataset(train_dataset, f"{DATA_PATH}/shape/train")
    save_dataset(val_dataset, f"{DATA_PATH}/shape/val")
    save_dataset(test_dataset, f"{DATA_PATH}/shape/test")


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
