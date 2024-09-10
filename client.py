import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:
    def __init__(self, model, criterion, client_loader, num_epochs=1):
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.train_loader = client_loader
        self.num_epochs = num_epochs

    def train(self, num_epochs=2):
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()

        for _ in range(num_epochs):
            for data, target in self.train_loader:
                data = data.view(data.size(0), 1, 28, 28)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

    def get_model_weights(self):
        return self.model.state_dict()

    def update_model_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
