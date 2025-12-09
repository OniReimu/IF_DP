import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, efficientnet_b0

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, _ = self.pool(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x, _ = self.pool(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x, _ = self.pool(x)
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNet18(nn.Module):
    """
    ResNet-18 model adapted for CIFAR-10 (10 classes).
    Uses torchvision's ResNet-18 as backbone.
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load pretrained ResNet-18 and modify for CIFAR-10
        resnet = resnet18(pretrained=False)
        
        # Modify first conv layer for CIFAR-10 (3 channels, 32x32 images)
        # Original ResNet expects 224x224, but we'll keep the same kernel size
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool since CIFAR-10 images are already small
        resnet.maxpool = nn.Identity()
        
        # Modify final layer for 10 classes (CIFAR-10)
        resnet.fc = nn.Linear(resnet.fc.in_features, 10)
        
        self.resnet = resnet
    
    def forward(self, x):
        return self.resnet(x)


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 adapted for CIFAR-10 (10 classes).
    Uses torchvision's EfficientNet-B0 as backbone.
    """
    def __init__(self):
        super().__init__()
        eff = efficientnet_b0(weights=None)
        # Adjust classifier for 10 classes
        in_features = eff.classifier[1].in_features
        eff.classifier[1] = nn.Linear(in_features, 10)
        self.efficientnet = eff

    def forward(self, x):
        return self.efficientnet(x)


def create_model(model_type='cnn'):
    """
    Factory function to create a model based on type.
    
    Args:
        model_type: 'cnn' for simple CNN, 'resnet18' for ResNet-18, 'efficientnet_b0' for EfficientNet-B0
    
    Returns:
        Model instance
    """
    if model_type.lower() == 'cnn':
        return CNN()
    elif model_type.lower() == 'resnet18' or model_type.lower() == 'resnet':
        return ResNet18()
    elif model_type.lower() in ['efficientnet_b0', 'efficientnet']:
        return EfficientNetB0()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn', 'resnet18', or 'efficientnet_b0'")