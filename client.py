import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ResNetCIFAR
from tqdm import tqdm
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Client:
    def __init__(self, client_loader, num_epochs=5, lr=0.001):
        self.model = ResNetCIFAR().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = client_loader
        self.num_epochs = num_epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # Reduce LR over time

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

            self.scheduler.step()  # Adjust learning rate

            # print(f'Epoch {epoch + 1} Loss: {running_loss / len(self.train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def get_model_weights(self):
        return self.model.state_dict()

    def update_model_weights(self, global_weights):
        self.model.load_state_dict(global_weights)