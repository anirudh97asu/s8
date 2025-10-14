import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=128):
    # Simple augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_data = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_data = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader