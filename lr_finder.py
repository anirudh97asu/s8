import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data import get_loaders
from model import ResNet34

class LRFinder:
    """
    Learning Rate Finder using exponential increase method
    Helps find optimal learning rate before training
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Store original state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()
    
    def find(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """
        Find learning rate by gradually increasing it
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
        """
        print("Finding optimal learning rate...")
        
        # Reset model to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        # Calculate learning rate multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        
        # Storage for results
        lrs = []
        losses = []
        best_loss = float('inf')
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        
        # Training loop
        self.model.train()
        iterator = iter(train_loader)
        
        for iteration in range(num_iter):
            # Get batch
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Store values
            current_lr = self.optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            losses.append(loss.item())
            
            # Check if loss is exploding
            if loss.item() > best_loss * 4:
                print(f"Stopping early at iteration {iteration}")
                break
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_mult
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{num_iter}, LR: {current_lr:.2e}, Loss: {loss.item():.4f}")
        
        # Reset model
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        return lrs, losses
    
    def plot(self, lrs, losses, skip_start=10, skip_end=5):
        """
        Plot learning rate vs loss
        
        Args:
            lrs: List of learning rates
            losses: List of losses
            skip_start: Skip first N points (noisy)
            skip_end: Skip last N points (exploding loss)
        """
        # Skip noisy start and exploding end
        lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
        losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
        
        # Find minimum loss
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]
        
        # Suggested LR is ~10x lower than min loss LR
        suggested_lr = min_loss_lr / 10
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, alpha=0.3)
        
        # Mark minimum and suggested
        plt.axvline(min_loss_lr, color='red', linestyle='--', 
                   label=f'Min Loss LR: {min_loss_lr:.2e}')
        plt.axvline(suggested_lr, color='green', linestyle='--', 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        
        plt.savefig('lr_finder_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as 'lr_finder_plot.png'")
        plt.show()
        
        print(f"\n{'='*50}")
        print(f"Minimum Loss LR: {min_loss_lr:.2e}")
        print(f"Suggested LR: {suggested_lr:.2e}")
        print(f"{'='*50}")
        
        return suggested_lr


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Data
    train_loader, _ = get_loaders(batch_size=128)
    
    # Model
    model = ResNet34(num_classes=100).to(device)
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=5e-4)
    
    # Find learning rate
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.find(
        train_loader, 
        start_lr=1e-7, 
        end_lr=10, 
        num_iter=100
    )
    
    # Plot and get suggestion
    suggested_lr = lr_finder.plot(lrs, losses)
    
    print(f"\nUse this learning rate in your training:")
    print(f"optimizer = optim.SGD(model.parameters(), lr={suggested_lr:.2e})")


if __name__ == '__main__':
    main()