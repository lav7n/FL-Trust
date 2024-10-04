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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

parser = argparse.ArgumentParser(description='Federated Learning with FLTrust and configurable parameters')
parser.add_argument('--num_clients', type=int, default=30, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=20, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=9, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for each client')
parser.add_argument('--FLTrust', action='store_true', help='Use FLTrust or not')
parser.add_argument('--attack_type', type=str, default='lr_poison', help='Type of attack to apply to malicious clients')
parser.add_argument('--noise_stddev', type=float, default=16, help='Standard deviation of noise for Gaussian noise attack')

args = parser.parse_args()


model = LeNet().to(device)
print("Model reset!")
criterion = nn.CrossEntropyLoss()


test = TestDataLoader(batch_size=64)
test_loader = test.get_test_loader()

client_data_loader = ClientDataLoader(num_clients=args.num_clients,
                                        num_malicious=args.num_malicious, 
                                        batch_size=64, 
                                        attack_type=args.attack_type, 
                                        noise_stddev=args.noise_stddev)
client_datasets = client_data_loader.get_client_datasets()

default_lr = 0.0001  # Default learning rate for non-malicious clients
malicious_lr = 0.1 if args.attack_type == 'lr_poison' else default_lr  # Poisoned LR for malicious clients if attack type is lr_poison
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
root_loader = RootClientDataLoader(batch_size=64)
root_client = Client(model=model, criterion=criterion, client_loader=root_loader.get_dataloader(), num_epochs=args.num_epochs)
server = Server(model, criterion, num_clients=args.num_clients, alpha=1)

print("FLTrust: ", args.FLTrust)
if args.FLTrust:
    print("FLTrust Enabled!")
    accuracies_with_fltrust, root_client_accuracies, A, B = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=True,
        root_client=root_client,
        root_client_only=False
    )
else:
    print("FLTrust Disabled!")
    accuracies_with_fltrust = server.Train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=False,
        root_client=None,
        root_client_only=False
    )
    root_client_accuracies = None

print("Global Model Accuracies across rounds with FLTrust:", accuracies_with_fltrust)
if root_client_accuracies:
    print("Root Client Accuracies across rounds:", root_client_accuracies)

PlotResult(
    accuracies_with_fltrust,
    root_accuracies=root_client_accuracies,
    fltrust_enabled=args.FLTrust,
    num_clients=args.num_clients,
    num_rounds=args.num_rounds,
    num_malicious=args.num_malicious
)

if args.FLTrust:

    def CosineSimilarityRowWise(A, B):
        """Computes cosine similarity between rows of matrices A and B."""
        num_rows = A.shape[0]
        C = np.zeros((num_rows, num_rows))

        # Compute row-wise cosine similarity
        for i in range(num_rows):
            for j in range(num_rows):
                dot_product = np.dot(A[i], B[j])
                norm1 = np.linalg.norm(A[i])
                norm2 = np.linalg.norm(B[j])

                # Store similarity, handling zero norms
                if norm1 != 0 and norm2 != 0:
                    C[i, j] = dot_product / (norm1 * norm2)
                else:
                    C[i, j] = 0  # Avoid NaN by setting similarity to 0 if any norm is zero

        return C

    def CosineSimilarityColumnWise(A, B):
        """Computes cosine similarity between columns of matrices A and B."""
        num_cols = A.shape[1]
        D = np.zeros((num_cols, num_cols))

        # Compute column-wise cosine similarity
        for i in range(num_cols):
            for j in range(num_cols):
                dot_product = np.dot(A[:, i], B[:, j])
                norm1 = np.linalg.norm(A[:, i])
                norm2 = np.linalg.norm(B[:, j])

                # Store similarity, handling zero norms
                if norm1 != 0 and norm2 != 0:
                    D[i, j] = dot_product / (norm1 * norm2)
                else:
                    D[i, j] = 0  # Avoid NaN by setting similarity to 0 if any norm is zero

        return D

    # Calculate row-wise and column-wise cosine similarity matrices
    C = CosineSimilarityRowWise(A, B)
    D = CosineSimilarityColumnWise(A, B)

    print("Row-wise Cosine Similarity Matrix C:\n", C)
    print("Column-wise Cosine Similarity Matrix D:\n", D)
