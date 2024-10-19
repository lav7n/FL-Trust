import os
import shutil
import argparse
import torch
import torch.nn as nn
from model import ResNetCIFAR  # Ensure you import the correct ResNet model
from dataloaders import DataLoaderManager
from server import Server
from client import Client
from tqdm import tqdm

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
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for clients')
parser.add_argument('--distribution', type=str, default='iid', help='Data distribution among clients')
args = parser.parse_args()

model = ResNetCIFAR().to(device)
criterion = nn.CrossEntropyLoss()

data_loader_manager = DataLoaderManager(batch_size=64, num_clients=args.num_clients, root_dataset_fraction=0.1, distribution=args.distribution)
test_loader = data_loader_manager.get_test_loader()
client_loaders = data_loader_manager.get_client_loaders()


default_lr = args.lr
malicious_lr = 1 if args.attack_type == 'lr_poison' else default_lr

clients = [
    Client(client_loader=train_loader, num_epochs=args.num_epochs, lr=(malicious_lr if i < args.num_malicious else default_lr))
    for i, train_loader in enumerate(client_loaders)
]

print(f"Number of clients created: {len(clients)}")
print(f"Total number of clients: {len(client_loaders)}")
print(f"Number of malicious clients: {args.num_malicious}")

root_client = Client(client_loader=data_loader_manager.get_root_loader(), num_epochs=args.num_epochs, lr=default_lr)
server = Server(model=model, criterion=criterion, num_clients=args.num_clients, alpha=1, print_metrics=args.printmetrics)

print("FLTrust: ", args.FLTrust)
if args.FLTrust:
    print("FLTrust Enabled!")
    accuracies, root_client_accuracies = server.train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=True,
        root_client=root_client
    )
else:
    print("FedAvg Enabled!")
    accuracies = server.train(
        clients, test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=False,
        root_client=None
    )
    root_client_accuracies = None

print("Global Model Accuracies across rounds:", accuracies)

# if args.FLTrust:
#     save_matrices(A, B, C, args.attack_type, args.num_clients, args.num_malicious)
