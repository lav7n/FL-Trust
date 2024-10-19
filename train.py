import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataloaders import ClientDataLoader, RootClientDataLoader, TestDataLoader
from server import Server
from client import Client
from model import ResNet
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
from dataloaders import save_matrices

if os.path.exists('HistoRounds'):
    shutil.rmtree('HistoRounds')
os.makedirs('HistoRounds')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

parser = argparse.ArgumentParser(description='Federated Learning with FLTrust and configurable parameters')
parser.add_argument('--num_clients', type=int, default=1, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=20, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=0, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for each client')
parser.add_argument('--FLTrust', action='store_true', help='Use FLTrust or not')
parser.add_argument('--attack_type', type=str, default='label_flipping', help='Type of attack to apply to malicious clients')
parser.add_argument('--noise_stddev', type=float, default=256, help='Standard deviation of noise for Gaussian noise attack')
parser.add_argument('--printmetrics', action='store_true', help='Print metrics or not')
args = parser.parse_args()


model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()

test = TestDataLoader(batch_size=64)
test_loader = test.get_test_loader()

client_data_loader = ClientDataLoader(num_clients=args.num_clients,
                                        num_malicious=args.num_malicious, 
                                        batch_size=64, 
                                        attack_type=args.attack_type, 
                                        noise_stddev=args.noise_stddev)
client_datasets = client_data_loader.get_client_datasets()

default_lr = 0.0001
malicious_lr = 1 if args.attack_type == 'lr_poison' else default_lr  

# Ensure client_datasets contains the expected number of clients
print(f"Total number of clients: {len(client_datasets)}")
print(f"Number of malicious clients: {args.num_malicious}")

clients = [
    Client(
        model=model,
        criterion=criterion,
        client_loader=train_loader,
        num_epochs=args.num_epochs,
        lr=(malicious_lr if i < args.num_malicious else default_lr)
    )
    for i, train_loader in enumerate(client_datasets)
]

print(f"Number of clients created: {len(clients)}")

root_loader = RootClientDataLoader(batch_size=64)
root_client = Client(model=model, criterion=criterion, client_loader=root_loader.get_dataloader(), num_epochs=args.num_epochs)
server = Server(model, criterion, num_clients=args.num_clients, alpha=1, printmetrics=args.printmetrics)

print("FLTrust: ", args.FLTrust)
if args.FLTrust:
    print("FLTrust Enabled!")
    accuracies, root_client_accuracies,A,B,C = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=True,
        root_client=root_client,
        root_client_only=False
    )
else:
    print("FedAvg")
    accuracies = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=False,
        root_client=None,
        root_client_only=False
    )
    root_client_accuracies = None

print("Global Model Accuracies across rounds:", accuracies)


if args.FLTrust:
    save_matrices(A, B, C, args.attack_type, args.num_clients, args.num_malicious)
