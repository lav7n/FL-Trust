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
import shutil
from datetime import datetime

if os.path.exists('HistoRounds'):
    shutil.rmtree('HistoRounds')
os.makedirs('HistoRounds')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

parser = argparse.ArgumentParser(description='Federated Learning with FLTrust and configurable parameters')
parser.add_argument('--num_clients', type=int, default=30, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=20, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=9, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for each client')
parser.add_argument('--FLTrust', action='store_true', help='Use FLTrust or not')
parser.add_argument('--attack_type', type=str, default='gaussian_noise', help='Type of attack to apply to malicious clients')
parser.add_argument('--noise_stddev', type=float, default=256, help='Standard deviation of noise for Gaussian noise attack')
args = parser.parse_args()


model = LeNet().to(device)
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
server = Server(model, criterion, num_clients=args.num_clients, alpha=1)

print("FLTrust: ", args.FLTrust)
if args.FLTrust:
    print("FLTrust Enabled!")
    accuracies, root_client_accuracies,A,B,C = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=True,
        root_client=root_client,
        root_client_only=False,
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

    print("Matrix A: ", A)
    print("Matrix B: ", B)
    print("Matrix C: ", C)

    # Plot matrix A
    plt.figure(figsize=(6, 6))  # Create a new figure for Matrix A
    im_A = plt.imshow(A, cmap='magma', interpolation='none')
    plt.title('Root model on Clients data')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', color='white', fontsize=4)
    plt.colorbar(im_A)  # Add colorbar to Matrix A
    plt.show()

    # Plot matrix B
    plt.figure(figsize=(6, 6))  # Create a new figure for Matrix B
    im_B = plt.imshow(B, cmap='magma', interpolation='none')
    plt.title('Client Models on Root Data')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            plt.text(j, i, f'{B[i, j]:.1f}', ha='center', va='center', color='white', fontsize=4)
    plt.colorbar(im_B)  # Add colorbar to Matrix B
    plt.show()

    # Prepare and plot matrix C
    C = np.clip(C, -1, 1)  # Ensure values are within [-1, 1]
    C = np.round(C, 2)     # Round values to 2 decimal points

    plt.figure(figsize=(6, 6))  # Create a new figure for Matrix C
    im_C = plt.imshow(C, cmap='magma', interpolation='none')
    plt.title('Cosine Matrix')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.colorbar(im_C)  # Add colorbar to Matrix C
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', color='white', fontsize=4)
    plt.show()

