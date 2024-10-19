import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCIFAR(nn.Module):
    def __init__(self):
        super(ResNetCIFAR, self).__init__()
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  

    def forward(self, x):
        return self.model(x)