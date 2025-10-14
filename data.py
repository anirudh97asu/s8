import cv2
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)  # PIL -> np.array (H, W, C) in RGB

        if self.transform is not None:
            image = self.transform(image=image)["image"]  # -> torch.FloatTensor (C, H, W)

        return image, label


def get_loaders(batch_size=128, num_workers=4):
    """
    CIFAR-100 dataloaders with pad+crop, flip, ColorJitter, cutout(CoarseDropout), normalize.
    """
    # CIFAR-100 channel stats -> Befor Augmentations
    #mean = (0.5071, 0.4867, 0.4408)
    #std  = (0.2675, 0.2565, 0.2761)

    mean = (0.49164050817489624, 0.4708925783634186, 0.4287804961204529)
    std = (0.27113330364227295, 0.26383477449417114, 0.27615490555763245)

    train_transform = A.Compose([
        # pad 4 then random crop back to 32x32 (Albumentations version)
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(height=32, width=32),

        A.HorizontalFlip(p=0.5),

        # Color jitter (roughly torchvision defaults but a bit milder)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.8
        ),

        # Cutout-style regularization
        A.CoarseDropout(
            max_holes=1,
            max_height=8,  # slightly stronger than 8
            max_width=8,
            fill_value=(0, 0, 0),
            p=0.5
        ),

        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    train_base = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_base  = datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    train_data = AlbumentationsDataset(train_base, train_transform)
    test_data  = AlbumentationsDataset(test_base, test_transform)

    pin = True  # safe on CPU, helpful on CUDA
    pw = num_workers > 0

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pw
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pw
    )

    return train_loader, test_loader
