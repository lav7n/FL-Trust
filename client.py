import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Client:
    def __init__(self, model, criterion, client_loader, num_epochs=1, lr=0.0001):
        self.model = copy.deepcopy(model).to(device)
        self.criterion = criterion
        self.train_loader = client_loader
        self.num_epochs = num_epochs
        self.lr = lr
    
    def train(self, num_epochs=2):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Adam for adaptive learning rate
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR over time
        self.model.train()

        for _ in range(num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), 3, 32, 32)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

            scheduler.step()  # Adjust learning rate

            print(f'Epoch Loss: {running_loss / len(self.train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    def get_model_weights(self):
        return self.model.state_dict()

    def update_model_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
