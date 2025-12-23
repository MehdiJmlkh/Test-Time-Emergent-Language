from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import torch
import shapeworld
from PIL import Image
import math
import datasets
import os
import pickle


class ObjectsDataset(Dataset):
    """
    Dataset that generates one-hot encoded attribute vectors for all possible combinations.
    Each sample represents a n-attribute object where each attribute can take d different values.
    """
    
    def __init__(self, num_attributes=4, num_values=10):
        self.num_attributes = num_attributes
        self.num_values = num_values

    def __len__(self):
        return self.num_values ** self.num_attributes

    def __getitem__(self, idx):
        """
        Generate one-hot encoded representation for the given index.
        
        Args:
            idx: Index representing a unique combination of attribute values
            
        Returns:
            tuple: (concatenated one-hot vectors, original index)
        """
        # Create one-hot vectors for each attribute
        attributes = []
        temp_idx = idx
        
        for _ in range(self.num_attributes):
            attribute_vector = torch.zeros(self.num_values)
            attribute_vector[temp_idx % self.num_values] = 1
            attributes.append(attribute_vector)
            temp_idx //= self.num_values
        
        # Concatenate all attribute vectors
        combined_attributes = torch.cat(attributes)
        
        return combined_attributes, idx


class DSprites(Dataset):
    def __init__(self, n=None, npz_path="/home/shared/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
        data = np.load(npz_path, allow_pickle=True)
        imgs = data["imgs"]
        if n != None:
            idx = torch.randperm(imgs.shape[0])[:n]
            self.imgs = imgs[idx]
        else:
            self.imgs = imgs
    
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32).unsqueeze(0)
        return img.permute(1, 2, 0), idx


class ShapeWorld(Dataset):
    def __init__(
        self, 
        n=128, 
        mode='train', 
        dtype='agreement', 
        name='existential', 
        collision_tolerance=0.0, 
        config='/home/shared/ShapeWorld/configs/agreement/existential/oneshape_test.json',
        **kwargs
    ):
        dataset = shapeworld.Dataset.create(dtype=dtype, name=name, collision_tolerance=collision_tolerance, config=config, **kwargs)
        generated = dataset.generate(n=n, mode=mode, include_model=True)
        self.imgs = generated['world']
    
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.imgs[idx]), idx
    

class CelebA(Dataset):
    def __init__(self, n=202599, path="/home/shared/data/celeba_cls/celeba_dino_cls.pt"):
        self.features = torch.load(path)
        self.features = self.features[:n]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        
        return self.features[idx], idx

class MNIST(Dataset):
    def __init__(self, mode='train', path="/home/shared/data/MNIST1"):
        self.data = torch.load(path + f"/{mode}.pt")        
        
    def __len__(self):
        return len(self.data["images"])
    
    def __getitem__(self, idx): 
        return self.data["images"][idx].permute(1, 2, 0), self.data["labels"][idx]

def generate_shape(n=10000, train_split=0.8, val_split=0.1, test_split=0.1):
    def save_dataset(dataset, path):
        os.makedirs(path, exist_ok=True)
        data = []
        for example in dataset:
            data.append(example)
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)


    train_dataset = datasets.ShapeWorld(n=math.floor(n * train_split), mode='train')
    val_dataset = datasets.ShapeWorld(n=math.floor(n * val_split), mode='validation')
    test_dataset = datasets.ShapeWorld(n=math.floor(n * test_split), mode='test')


    save_dataset(train_dataset, "/home/shared/data/shape/train")
    save_dataset(val_dataset, "/home/shared/data/shape/val")
    save_dataset(test_dataset, "/home/shared/data/shape/test")
