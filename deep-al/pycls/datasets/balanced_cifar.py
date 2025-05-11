import numpy as np
import random
from pycls.datasets.custom_datasets import CIFAR10, CIFAR100
from PIL import Image

class BALANCEDCIFAR10(CIFAR10):
    cls_num = 10

    def __init__(self, root, train, transform=None, test_transform=None, download=True, only_features=False, samples_per_class=100):
        super(BALANCEDCIFAR10, self).__init__(root, train, transform=transform, test_transform=test_transform, download=download)
        self.train = train
        self.transform = transform
        self.test_transform = test_transform
        self.only_features = only_features
        self.samples_per_class = samples_per_class

        if self.train:
            self.gen_balanced_data()
            phase = 'Train'
        else:
            phase = 'Test'
        self.labels = self.targets

        print(f"{phase} Mode: Contain {len(self.data)} images")

    def gen_balanced_data(self):
        """Randomly balance the dataset to match the `_balance_dataset` logic."""
        class_counts = {cls: 0 for cls in range(self.cls_num)}
        balanced_data = []
        balanced_targets = []
        max_samples_per_class = self.samples_per_class

        random.seed(42)  # Set the random seed for reproducibility
        indices = list(range(len(self.data)))
        random.shuffle(indices)  # Shuffle globally like `_balance_dataset`

        for idx in indices:
            target = self.targets[idx]
            if class_counts[target] < max_samples_per_class:
                balanced_data.append(self.data[idx])
                balanced_targets.append(target)
                class_counts[target] += 1

            if all(count >= max_samples_per_class for count in class_counts.values()):
                break

        self.data = np.array(balanced_data)
        self.targets = balanced_targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # Convert to a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_cls_num_list(self):
        return [self.samples_per_class] * self.cls_num
