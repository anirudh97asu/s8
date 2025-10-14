import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from data import get_loaders
from model import ResNet34


class LRFinder:
    """
    Learning Rate Finder using exponential increase.
    Fixes:
      - Deep copy of model/optimizer states to truly restore initial weights/buffers.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Deep-copy original states (so running stats / momentum don't mutate under the hood)
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

    @torch.no_grad()
    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def find(self, train_loader, start_lr=1e-7, end_lr=1.0, num_iter=100, stop_factor=4.0):
        """
        Gradually increase LR from start_lr to end_lr over num_iter iterations.

        Args:
            train_loader: data loader
            start_lr: starting LR
            end_lr: ending LR (cap it reasonably to avoid nuking BN stats; 1.0 is a good default)
            num_iter: number of iterations (batches) to run
            stop_factor: stop early if loss > best_loss * stop_factor
        """
        print("Finding optimal learning rate...")

        # Reset model/optimizer to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # LR multiplier per iteration (exponential schedule)
        lr_mult = (end_lr / start_lr) ** (1.0 / max(1, num_iter))
        self._set_lr(start_lr)

        lrs, losses = [], []
        best_loss = float("inf")

        self.model.train()
        iterator = iter(train_loader)

        # optional: use torch.set_grad_enabled for clarity
        for it in range(num_iter):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward/backward
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss_value = loss.item()

            # Record
            current_lr = self.optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            losses.append(loss_value)

            # Track best and stop on explosion
            if loss_value < best_loss:
                best_loss = loss_value
            if loss_value > best_loss * stop_factor:
                print(f"Stopping early at iteration {it+1} due to exploding loss.")
                break

            loss.backward()
            self.optimizer.step()

            # Bump LR
            self._set_lr(current_lr * lr_mult)

            if (it + 1) % 10 == 0:
                print(f"Iter {it+1:3d}/{num_iter} | LR: {current_lr:.2e} | Loss: {loss_value:.4f}")

        # Restore initial weights/optimizer
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        return lrs, losses

    def plot(self, lrs, losses, skip_start=10, skip_end=5, suggest_cap=0.1):
        """
        Plot LR vs Loss and return a suggested LR (~10x below min-loss LR, with a safety cap).

        Args:
            lrs: list of learning rates
            losses: list of losses
            skip_start: ignore the first N points (noisy warmup)
            skip_end: ignore the last N points (post-explosion tail)
            suggest_cap: upper bound for the suggested LR (useful sanity cap)
        """
        # Guard against short runs
        if len(lrs) <= (skip_start + max(0, skip_end)):
            skip_start = 0
            skip_end = 0

        lrs_plot = lrs[skip_start: len(lrs) - skip_end if skip_end > 0 else None]
        losses_plot = losses[skip_start: len(losses) - skip_end if skip_end > 0 else None]

        # Find LR at minimum loss
        min_idx = int(np.argmin(losses_plot))
        min_loss_lr = float(lrs_plot[min_idx])

        suggested_lr = min(min_loss_lr / 10.0, suggest_cap)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs_plot, losses_plot)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, alpha=0.3)

        plt.axvline(min_loss_lr, linestyle="--", label=f"Min Loss LR: {min_loss_lr:.2e}")
        plt.axvline(suggested_lr, linestyle="--", label=f"Suggested LR: {suggested_lr:.2e}")
        plt.legend()

        plt.savefig("lr_finder_plot.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved as 'lr_finder_plot.png'")
        plt.show()

        print("\n" + "=" * 50)
        print(f"Minimum Loss LR: {min_loss_lr:.2e}")
        print(f"Suggested LR   : {suggested_lr:.2e}  (capped at {suggest_cap})")
        print("=" * 50)

        return suggested_lr


def main():
    # Speed tweak for fixed image sizes on CUDA:
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Data
    train_loader, _ = get_loaders(batch_size=128, num_workers=4)

    # Model
    model = ResNet34(num_classes=100).to(device)

    # Optimizer/Loss (match your training hyperparams)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=5e-4)

    # LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.find(
        train_loader,
        start_lr=1e-7,
        end_lr=1.0,     # safer than 10.0
        num_iter=100,
        stop_factor=4.0
    )

    # Plot & suggestion
    suggested_lr = lr_finder.plot(lrs, losses, skip_start=10, skip_end=5, suggest_cap=0.1)

    print("\nUse this learning rate in your training:")
    print(f"optimizer = optim.SGD(model.parameters(), lr={suggested_lr:.2e}, momentum=0.9, weight_decay=5e-4)")


if __name__ == "__main__":
    main()
