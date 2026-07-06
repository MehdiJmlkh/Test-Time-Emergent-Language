from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import torch
import shapeworld # type: ignore
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
        config="configs/one_shape.json",
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


def generate_shape(data_path, n=20000, train_split=0.8, val_split=0.1, test_split=0.1):
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
    test_dataset_two_shape = ShapeWorld(
        n=math.floor(n * test_split), mode="test", config="configs/two_shape.json"
    )

    save_dataset(train_dataset, f"{data_path}/shape/train")
    save_dataset(val_dataset, f"{data_path}/shape/val")
    save_dataset(test_dataset, f"{data_path}/shape/test/one_shape")
    save_dataset(test_dataset_two_shape, f"{data_path}/shape/test/two_shape")
