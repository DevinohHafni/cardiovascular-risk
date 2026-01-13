import torch.nn as nn
from torchvision import models

def get_backbone():
    model = models.resnet18(weights="IMAGENET1K_V1")
    features = model.fc.in_features
    model.fc = nn.Identity()
    return model, features

