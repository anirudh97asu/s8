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
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label


def get_loaders(batch_size=512):
    train_transform = A.Compose([
        A.RandomCrop(height=32, width=32, padding=4),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.5),
        A.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
        ToTensorV2()
    ])
    
    train_data_base = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
    test_data_base = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)
    
    train_data = AlbumentationsDataset(train_data_base, train_transform)
    test_data = AlbumentationsDataset(test_data_base, test_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader