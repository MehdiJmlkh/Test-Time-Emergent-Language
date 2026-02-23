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
from solve_min_sym import min_m_controlled_sampling
        
class ObjectsDataset(Dataset):
    def __init__(self, num_attributes=4, num_values=10, indices=None, min_symbol=2, batch_size=32, number_of_samples=None):
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.data = dict(enumerate(product(range(num_values), repeat=num_attributes)))
        
        if indices is None:
            indices = list(self.data.keys())
        
        self.data = {i: self.data[k] for i, k in enumerate(indices)}
        
        self.candidates = min_m_controlled_sampling(
            [(self.data[k], k) for k in self.data.keys()],
            number_of_samples=number_of_samples,
            target_min_symbol=min_symbol,
            batch_size=batch_size
        )

        self.samples = []
        for target, cand in self.candidates.items():
            self.samples.append((cand, cand.index(target)))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cands, target_idx = self.samples[idx]
        one_hots = torch.stack([self.__to_one_hot(torch.tensor(cand)) for cand in cands])
        return one_hots, target_idx
    
    def get_candidates(self):
        return self.candidates
    
    def __to_one_hot(self, x):
        one_hot = F.one_hot(x, num_classes=self.num_values)
        return one_hot.flatten().float()


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
        return torch.tensor(self.imgs[idx]), torch.tensor(1).to(device='cuda')
    

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
        return self.data["images"][idx].permute(1, 2, 0), torch.tensor(1).to(device='cuda')

def generate_shape(n=10000, train_split=0.8, val_split=0.1, test_split=0.1):
    def save_dataset(dataset, path):
        os.makedirs(path, exist_ok=True)
        data = []
        for example in dataset:
            data.append(example)
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)


    train_dataset = ShapeWorld(n=math.floor(n * train_split), mode='train')
    val_dataset = ShapeWorld(n=math.floor(n * val_split), mode='validation')
    test_dataset = ShapeWorld(n=math.floor(n * test_split), mode='test')


    save_dataset(train_dataset, "/home/shared/data/shape/train")
    save_dataset(val_dataset, "/home/shared/data/shape/val")
    save_dataset(test_dataset, "/home/shared/data/shape/test")
    
    
class PreSavedBatchDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]
