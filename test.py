import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torchvision
import torch
import datetime

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


path = 'weights/14_37_01/weights_keyboard_interrupt_epoch_1207_loss_0.0001627.pth'

# Code Alexnet Model from scratch
class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), # 55 x 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27 x 27
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 27 x 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 13 x 13
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 13 x 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 13 x 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 13 x 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 6 x 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AlexNet(num_classes=4)
model.load_state_dict(torch.load(path))

# Set the model to evaluation mode
model.eval()


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((227,227)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


# Load the test dataset
test_dataset = torchvision.datasets.ImageFolder(
    root='Dataset/test',
    transform= transform_train
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)


# Test the model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the accuracy of the model on the test dataset
print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

