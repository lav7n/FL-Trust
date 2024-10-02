import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Client:
    def __init__(self, model, criterion, client_loader, num_epochs=1):
        # Move the model to the appropriate device
        self.model = copy.deepcopy(model).to(device)
        self.criterion = criterion
        self.train_loader = client_loader
        self.num_epochs = num_epochs

    def train(self, num_epochs=2):
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.model.train()

        for _ in range(num_epochs):
            for data, target in self.train_loader:
                # Move data and target to the appropriate device
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), 1, 28, 28)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

    def get_model_weights(self):
        return self.model.state_dict()

    def update_model_weights(self, global_weights):
        # Load global weights onto the model (which is already on the correct device)
        self.model.load_state_dict(global_weights)
