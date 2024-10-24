import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Client:
    def __init__(self, client_loader, num_epochs=5, lr=0.001, in_channels=1, num_classes=1):
        # Initialize the U-Net model from segmentation_models_pytorch (smp)
        self.model = smp.Unet(
            encoder_name="resnet34",        # Use ResNet-34 as the encoder
            encoder_weights="imagenet",     # Pretrained on ImageNet
            in_channels=in_channels,        # Number of input channels (1 for grayscale images)
            classes=num_classes             # Number of output classes (1 for binary segmentation)
        ).to(device)

        # Loss function for binary segmentation (you can switch to DiceLoss if needed)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # Reduce LR over time

        self.train_loader = client_loader
        self.num_epochs = num_epochs

    def train(self, fedprox=False, mu=0.0, global_weights=None):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Apply FedProx proximal term if enabled
                if fedprox and global_weights is not None:
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_weights.values()):
                        proximal_term += ((param - global_param.to(device)) ** 2).sum()
                    loss += (mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            self.scheduler.step()  # Adjust learning rate after each epoch

    def evaluate(self, test_loader):
        self.model.eval()
        iou_score = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                
                # Threshold outputs for binary mask prediction
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                # Calculate IoU (Intersection over Union) as a metric for segmentation
                intersection = (preds * target).sum((1, 2, 3))
                union = (preds + target).sum((1, 2, 3)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero

                iou_score += iou.mean().item()
                total += 1

        return iou_score / total  # Return the average IoU across the test set

    def get_model_weights(self):
        return self.model.state_dict()

    def update_model_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
