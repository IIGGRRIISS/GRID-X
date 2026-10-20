import torch
import torch.nn as nn
from torchvision import models
import numpy as np

def get_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def extract_features(model, dataloader, device):
    """
    Extract feature vectors from the penultimate layer.
    Returns:
        features: numpy array (n_samples, 2048)
        labels: list of class indices
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            # Forward pass up to avgpool
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            features.append(x.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    features = np.vstack(features)
    return features, labels