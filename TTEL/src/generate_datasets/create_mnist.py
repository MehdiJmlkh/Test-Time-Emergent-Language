import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import os


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


def create_mnist1(data_path):
    mnist_path = f"{data_path}/MNIST1"
    train_file = f"{mnist_path}/train.pt"
    test_file = f"{mnist_path}/test.pt"

    # ✅ Skip everything if both already exist
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"[SKIP] MNIST1 already exists in {mnist_path}")
        return

    os.makedirs(mnist_path, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )

    singleDigit_train = SingleDigitPadded(train_dataset)
    singleDigit_test = SingleDigitPadded(test_dataset)

    train_loader = DataLoader(singleDigit_train, batch_size=len(singleDigit_train))
    test_loader = DataLoader(singleDigit_test, batch_size=len(singleDigit_test))

    for images, labels, positions in train_loader:
        torch.save(
            {
                "images": images,
                "labels": labels,
                "positions": positions,
            },
            train_file,
        )
        break

    for images, labels, positions in test_loader:
        torch.save(
            {
                "images": images,
                "labels": labels,
                "positions": positions,
            },
            test_file,
        )
        break


def create_mnist2(data_path):
    mnist_path = f"{data_path}/MNIST2"
    output_file = f"{mnist_path}/test.pt"

    if os.path.exists(output_file):
        print(f"[SKIP] MNIST2 already exists in {mnist_path}")
        return

    os.makedirs(mnist_path, exist_ok=True)

    singleDigit = torch.load(f"{data_path}/MNIST1/test.pt")

    two_digit_dataset = TwoDigitOpposite(singleDigit)

    loader = DataLoader(two_digit_dataset, batch_size=len(two_digit_dataset))

    for images, labels in loader:
        torch.save(
            {
                "images": images,  # Tensor [N, 1, 28, 56]
                "labels": labels[0] * 10 + labels[1],
            },
            output_file,
        )
        break
