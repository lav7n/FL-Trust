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
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Federated Learning with FLTrust and configurable parameters')

# Adding arguments for num_clients, num_rounds, num_malicious, num_epochs, and FLTrust
parser.add_argument('--num_clients', type=int, default=30, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=20, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=0, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for each client')
parser.add_argument('--FLTrust', type=bool, default=True, help='Use FLTrust or not')

args = parser.parse_args()

# Initialize the model and loss criterion
model = LeNet()
print("Model reset!")
criterion = nn.CrossEntropyLoss()

# Initialize the test dataloader
test = TestDataLoader(batch_size=64)
test_loader = test.get_test_loader()

# Load the client datasets based on the argparse values
client_data_loader = ClientDataLoader(num_clients=args.num_clients, num_malicious=args.num_malicious, batch_size=64, attack_type='label_flipping')
client_datasets = client_data_loader.get_client_datasets()

# Create clients using the parsed number of epochs
clients = [Client(model=model, criterion=criterion, client_loader=train_loader, num_epochs=args.num_epochs)
           for train_loader in client_datasets]

# Initialize the root client
root_loader = RootClientDataLoader(batch_size=64)
root_client = Client(model=model, criterion=criterion, client_loader=root_loader.get_dataloader(), num_epochs=args.num_epochs)

# Initialize the server with the parsed number of clients
server = Server(model, criterion, num_clients=args.num_clients, alpha=1)

# Train the model using the provided arguments
accuracies_with_fltrust, root_client_accuracies = server.Train(
    clients, test_loader,
    num_rounds=args.num_rounds,
    num_epochs=args.num_epochs,
    FLTrust=args.FLTrust,
    root_client=root_client,
    root_client_only=False
)

# Output the results
print("Global Model Accuracies across rounds with FLTrust:", accuracies_with_fltrust)
print("Root Client Accuracies across rounds:", root_client_accuracies)