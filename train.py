import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataloaders import ClientDataLoader, RootClientDataLoader, TestDataLoader
from server import Server
from client import Client
from model import LeNet
import numpy as np

model = LeNet()
print("Model reset!")
criterion = nn.CrossEntropyLoss()

test = TestDataLoader(batch_size=64)
test_loader = test.get_test_loader()

client_data_loader = ClientDataLoader(num_clients=10, num_malicious=0, batch_size=64, attack_type='label_flipping')
client_datasets = client_data_loader.get_client_datasets()

clients = [Client(model=model, criterion=criterion, client_loader=train_loader, num_epochs=2)
           for train_loader in client_datasets]

root_loader = RootClientDataLoader(batch_size=64)
root_client = Client(model=model, criterion=criterion, client_loader=root_loader.get_dataloader(), num_epochs=2)
server = Server(model, criterion, num_clients=10, alpha=1)


accuracies_with_fltrust, root_client_accuracies = server.Train(
    clients, test_loader,
     num_rounds=10,
      num_epochs=2, 
      FLTrust=True, 
      root_client=root_client,
       root_client_only=False
)

print("Global Model Accuracies across rounds with FLTrust:", accuracies_with_fltrust)
print("Root Client Accuracies across rounds:", root_client_accuracies)
