import random
from torch.utils.data import Sampler


class OneClassBatchSampler(Sampler):
    def __init__(self, class_to_indices, num_classes=50, shuffle=True):
        """
        Args:
            class_to_indices: dict[class_label] -> list of sample indices
            num_classes: number of classes to use (<= total classes)
            shuffle: shuffle order of batches
        """
        self.shuffle = shuffle

        all_classes = list(class_to_indices.keys())
        if num_classes > len(all_classes):
            raise ValueError("num_classes cannot exceed total number of classes")

        random.Random().shuffle(all_classes)
        selected_classes = all_classes[:num_classes]

        # Keep only selected classes
        self.class_batches = [class_to_indices[c] for c in selected_classes]

    def __iter__(self):
        if self.shuffle:
            random.Random().shuffle(self.class_batches)
        for batch in self.class_batches:
            yield batch

    def __len__(self):
        return len(self.class_batches)


class DogClassBatchSampler(Sampler):
    def __init__(self, class_to_indices, num_classes=50, shuffle=True, batch_size=32):
        """
        Args:
            class_to_indices: dict[class_label] -> list of sample indices
            num_classes: number of classes to use (<= total classes)
            shuffle: shuffle order of batches
        """
        self.shuffle = shuffle
        self.batch_size = batch_size

        all_classes = sorted(list(class_to_indices.keys()))
        if num_classes > len(all_classes):
            raise ValueError("num_classes cannot exceed total number of classes")

        dog_classes = all_classes[151:269]
        random.shuffle(dog_classes)
        selected_classes = dog_classes[:batch_size]
        # Keep only selected classes
        self.class_batches = [class_to_indices[c] for c in selected_classes]

    def __iter__(self):
        for i in range(32):
            yield [class_batch[i] for class_batch in self.class_batches]

    def __len__(self):
        return 32


class PreSavedBatchSampler(Sampler):
    def __init__(self, number_batches=9):
        self.batch_indices = [i for i in range(number_batches)]

    def __iter__(self):
        for idx in self.batch_indices:
            yield [idx]

    def __len__(self):
        return len(self.batch_indices)
