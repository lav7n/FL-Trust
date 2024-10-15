import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load the pre-trained ResNet model and modify it for CIFAR-10
        self.model = models.resnet18(weights=None)  # Set weights=None to avoid loading pre-trained weights
        # Change the first convolution layer to accept 3 channels (CIFAR-10 images are RGB)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Update the fully connected layer for 10 output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)
