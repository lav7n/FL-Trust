import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataloaders import ClientDataLoader, RootClientDataLoader, TestDataLoader, PlotResult
from server import Server
from client import Client
from model import LeNet
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

parser = argparse.ArgumentParser(description='Federated Learning with FLTrust and configurable parameters')
parser.add_argument('--num_clients', type=int, default=15, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=10, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=5, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for each client')
parser.add_argument('--FLTrust', type=bool, default=True, help='Use FLTrust or not')

args = parser.parse_args()

# Initialize the model and move it to the appropriate device
model = LeNet().to(device)
print("Model reset!")
criterion = nn.CrossEntropyLoss()

# Load the test data
test = TestDataLoader(batch_size=64)
test_loader = test.get_test_loader()

# Load client data
client_data_loader = ClientDataLoader(num_clients=args.num_clients, num_malicious=args.num_malicious, batch_size=64, attack_type='label_flipping')
client_datasets = client_data_loader.get_client_datasets()

# Create clients and move their models to the appropriate device
clients = [Client(model=model, criterion=criterion, client_loader=train_loader, num_epochs=args.num_epochs)
           for train_loader in client_datasets]

# Create the root client and move it to the appropriate device
root_loader = RootClientDataLoader(batch_size=64)
root_client = Client(model=model, criterion=criterion, client_loader=root_loader.get_dataloader(), num_epochs=args.num_epochs)

# Initialize the server
server = Server(model, criterion, num_clients=args.num_clients, alpha=1)

# Start training
if args.FLTrust:
    accuracies_with_fltrust, root_client_accuracies = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=args.FLTrust,
        root_client=root_client,
        root_client_only=False
    )
else:
    accuracies_with_fltrust = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=args.FLTrust,
        root_client=None,
        root_client_only=False
    )
    root_client_accuracies = None

print("Global Model Accuracies across rounds with FLTrust:", accuracies_with_fltrust)
if root_client_accuracies:
    print("Root Client Accuracies across rounds:", root_client_accuracies)


PlotResult(accuracies_with_fltrust, root_client_accuracies, fltrust_enabled=args.FLTrust)
