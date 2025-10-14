import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-34
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU
    """
    expansion = 1  # Output channels = input channels * expansion
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection (identity or projection)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut path
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add shortcut to main path
        out += identity
        out = self.relu(out)
        
        return out


class ResNet34(nn.Module):
    """
    ResNet-34 Architecture for CIFAR-100
    
    Architecture Overview:
    - Initial Conv Layer (64 filters)
    - Layer 1: 3 BasicBlocks, 64 channels  (output: 32x32)
    - Layer 2: 4 BasicBlocks, 128 channels (output: 16x16)
    - Layer 3: 6 BasicBlocks, 256 channels (output: 8x8)
    - Layer 4: 3 BasicBlocks, 512 channels (output: 4x4)
    - Global Average Pooling
    - Fully Connected Layer (100 classes)
    
    Total: 3 + 4 + 6 + 3 = 16 BasicBlocks = 32 conv layers + 1 initial + 1 fc = 34 layers
    """
    
    def __init__(self, num_classes=100):
        super(ResNet34, self).__init__()
        
        self.in_channels = 64
        
        # ============ Initial Convolution Layer ============
        # Input: 3x32x32 -> Output: 64x32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Note: No maxpool for CIFAR (images are only 32x32)
        # ImageNet ResNet uses: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ============ Residual Layers ============
        # Layer 1: 3 blocks, 64 channels, stride=1
        # Output: 64x32x32
        self.layer1 = self._make_layer(
            out_channels=64, 
            num_blocks=3, 
            stride=1
        )
        
        # Layer 2: 4 blocks, 128 channels, stride=2
        # Output: 128x16x16
        self.layer2 = self._make_layer(
            out_channels=128, 
            num_blocks=4, 
            stride=2
        )
        
        # Layer 3: 6 blocks, 256 channels, stride=2
        # Output: 256x8x8
        self.layer3 = self._make_layer(
            out_channels=256, 
            num_blocks=6, 
            stride=2
        )
        
        # Layer 4: 3 blocks, 512 channels, stride=2
        # Output: 512x4x4
        self.layer4 = self._make_layer(
            out_channels=512, 
            num_blocks=3, 
            stride=2
        )
        
        # ============ Final Layers ============
        # Global Average Pooling: 512x4x4 -> 512x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer: 512 -> num_classes
        self.fc = nn.Linear(512, num_classes)
        
        # ============ Weight Initialization ============
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Create a residual layer with multiple BasicBlocks
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of BasicBlocks in this layer
            stride: Stride for the first block (for downsampling)
        """
        downsample = None
        
        # If dimensions change, we need a projection shortcut
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(
            BasicBlock(
                self.in_channels, 
                out_channels, 
                stride, 
                downsample
            )
        )
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Remaining blocks (no downsampling)
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    self.in_channels, 
                    out_channels
                )
            )
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Input shape: (batch_size, 3, 32, 32)
        Output shape: (batch_size, num_classes)
        """
        # Initial convolution
        # (B, 3, 32, 32) -> (B, 64, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual layers
        # (B, 64, 32, 32) -> (B, 64, 32, 32)
        x = self.layer1(x)
        
        # (B, 64, 32, 32) -> (B, 128, 16, 16)
        x = self.layer2(x)
        
        # (B, 128, 16, 16) -> (B, 256, 8, 8)
        x = self.layer3(x)
        
        # (B, 256, 8, 8) -> (B, 512, 4, 4)
        x = self.layer4(x)
        
        # Global average pooling
        # (B, 512, 4, 4) -> (B, 512, 1, 1)
        x = self.avgpool(x)
        
        # Flatten
        # (B, 512, 1, 1) -> (B, 512)
        x = torch.flatten(x, 1)
        
        # Fully connected layer
        # (B, 512) -> (B, num_classes)
        x = self.fc(x)
        
        return x


def resnet34(num_classes=100):
    """Factory function to create ResNet-34 model"""
    return ResNet34(num_classes=num_classes)


# ============ Model Summary ============
if __name__ == '__main__':
    model = ResNet34(num_classes=100)
    
    # Print model structure
    print("=" * 60)
    print("ResNet-34 Architecture for CIFAR-100")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print layer-by-layer architecture
    print("\n" + "=" * 60)
    print("Layer-by-Layer Architecture:")
    print("=" * 60)
    print("\n1. Initial Conv Layer:")
    print(f"   Conv2d(3, 64, kernel_size=3, stride=1, padding=1)")
    print(f"   -> Output: 64x32x32")
    
    print("\n2. Layer 1 (3 BasicBlocks, 64 channels):")
    print(f"   -> Output: 64x32x32")
    
    print("\n3. Layer 2 (4 BasicBlocks, 128 channels):")
    print(f"   -> Output: 128x16x16 (stride=2 downsampling)")
    
    print("\n4. Layer 3 (6 BasicBlocks, 256 channels):")
    print(f"   -> Output: 256x8x8 (stride=2 downsampling)")
    
    print("\n5. Layer 4 (3 BasicBlocks, 512 channels):")
    print(f"   -> Output: 512x4x4 (stride=2 downsampling)")
    
    print("\n6. Global Average Pooling:")
    print(f"   -> Output: 512x1x1")
    
    print("\n7. Fully Connected Layer:")
    print(f"   Linear(512, 100)")
    print(f"   -> Output: 100 classes")
    
    print("\n" + "=" * 60)