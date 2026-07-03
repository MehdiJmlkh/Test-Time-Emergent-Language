import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random


class SingleDigitPadded(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]

        # Decide randomly whether the digit is on the left or right
        if random.random() < 0.5:
            # digit on left
            empty = torch.zeros_like(img)
            image = torch.cat((img, empty), dim=2)  # 28x56
            position = "left"
        else:
            # digit on right
            empty = torch.zeros_like(img)
            image = torch.cat((empty, img), dim=2)
            position = "right"

        return image, label, position


class TwoDigitOpposite(Dataset):
    def __init__(self, padded_dataset):
        """
        padded_dataset: your SingleDigitPadded dataset
        """
        self.images = padded_dataset["images"]
        self.positions = padded_dataset["positions"]
        self.labels = padded_dataset["labels"]

        # Precompute indices by position for faster sampling
        self.left_indices = [
            i for i in range(len(self.images)) if self.positions[i] == "left"
        ]
        self.right_indices = [
            i for i in range(len(self.images)) if self.positions[i] == "right"
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # First sample (fixed)
        img1, label1, pos1 = self.images[idx], self.labels[idx], self.positions[idx]

        # Determine opposite position
        opposite_pos = "right" if pos1 == "left" else "left"
        candidate_indices = (
            self.right_indices if opposite_pos == "right" else self.left_indices
        )

        # Randomly select second sample with opposite position
        idx2 = random.choice(candidate_indices)
        img2, label2 = self.images[idx2], self.labels[idx2]

        two_digit_img = img1 + img2
        two_digit_label = (label1, label2) if pos1 == "left" else (label2, label1)

        return two_digit_img, two_digit_label


def create_mnist1():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root="/home/shared/data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="/home/shared/data", train=False, download=True, transform=transform
    )

    singleDigit_train = SingleDigitPadded(train_dataset)
    singleDigit_test = SingleDigitPadded(test_dataset)

    train_loader = DataLoader(singleDigit_train, batch_size=len(singleDigit_train))
    test_loader = DataLoader(singleDigit_test, batch_size=len(singleDigit_test))

    for images, labels, positions in train_loader:
        torch.save(
            {
                "images": images,  # Tensor [N, 1, 28, 56]
                "labels": labels,
                "positions": positions,
            },
            "/home/shared/data/MNIST2/train.pt",
        )
        break  # only one batch needed

    for images, labels, positions in test_loader:
        torch.save(
            {
                "images": images,  # Tensor [N, 1, 28, 56]
                "labels": labels,
                "positions": positions,
            },
            "/home/shared/data/MNIST2/test.pt",
        )
        break  # only one batch needed


def create_mnist2():
    singleDigit = torch.load("/home/shared/data/MNIST1/test.pt")

    two_digit_dataset = TwoDigitOpposite(singleDigit)

    # Save the dataset
    loader = DataLoader(two_digit_dataset, batch_size=len(two_digit_dataset))

    for images, labels in loader:
        torch.save(
            {
                "images": images,  # Tensor [N, 1, 28, 56]
                "labels": labels[0] * 10 + labels[1],
            },
            "/home/shared/data/MNIST2/test.pt",
        )
        break
