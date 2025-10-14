import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
from model import ResNet34
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    train_loader, test_loader = get_loaders(batch_size=128)

    torch.set_float32_matmul_precision("high")
    
    # Model
    model = ResNet34(num_classes=100).to(device)
    model = torch.compile(model)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    epochs = 50
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = test(
            model, test_loader, criterion, device
        )
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'cifar100_model.pth')
    print('Model saved!')

if __name__ == '__main__':
    main()