import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from dataloaders import DataLoaderManager
from server import Server
from client import Client
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Argument parsing
parser = argparse.ArgumentParser(description='Federated Learning with FLTrust and configurable parameters')
parser.add_argument('--num_clients', type=int, default=50, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=15, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=10, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for each client')
parser.add_argument('--FLTrust', action='store_true', help='Use FLTrust or not')
parser.add_argument('--fedprox', action='store_true', help='Use FedProx or not')
parser.add_argument('--attack_type', type=str, default='gaussian', help='Type of attack to apply to malicious clients')
parser.add_argument('--noise_stddev', type=float, default=256, help='Standard deviation of noise for Gaussian noise attack')
parser.add_argument('--printmetrics', action='store_true', help='Print metrics or not')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for clients')
parser.add_argument('--distribution', type=str, default='iid', help='Data distribution among clients')
args = parser.parse_args()

# Initialize model and criterion
model = smp.Unet(
    encoder_name="resnet34",        # Encoder: you can change this to other backbones
    encoder_weights="imagenet",     # Pretrained on ImageNet
    in_channels=1,                  # Grayscale images for retinal segmentation
    classes=1                       # Binary segmentation (vessel vs background)
).to(device)

criterion = nn.BCEWithLogitsLoss()

# Initialize data loaders with potential attacks on malicious clients
data_loader_manager = DataLoaderManager(
    batch_size=4,  # Adjust batch size for segmentation task (may require lower value due to image size)
    num_clients=args.num_clients, 
    root_dataset_fraction=0.1, 
    distribution=args.distribution,
    num_malicious=args.num_malicious,
    attack_type=args.attack_type,
    noise_stddev=args.noise_stddev
)

# Get test loader and client loaders
# test_loader = data_loader_manager.get_test_loader()
client_loaders = data_loader_manager.get_client_loaders()

# Learning rates for clients, modified for malicious clients if required
default_lr = args.lr
malicious_lr = 1 if args.attack_type == 'lr_poison' else default_lr

# Initialize clients, distinguishing between malicious and benign clients
clients = [
    Client(client_loader=train_loader, num_epochs=args.num_epochs, lr=(malicious_lr if i < args.num_malicious else default_lr))
    for i, train_loader in enumerate(client_loaders)
]

print(f"Number of clients created: {len(clients)}")
print(f"Total number of clients: {len(client_loaders)}")
print(f"Number of malicious clients: {args.num_malicious}")

# Root client setup for FLTrust
root_client = Client(client_loader=data_loader_manager.get_root_loader(), num_epochs=args.num_epochs, lr=default_lr)

# Server initialization
server = Server(model=model, criterion=criterion, num_clients=args.num_clients, alpha=1, mu=args.fedprox, print_metrics=args.printmetrics)

# FLTrust vs FedAvg training loop
print("FLTrust: ", args.FLTrust)
print("FedProx: ", args.fedprox)
if args.FLTrust:
    print("FLTrust Enabled!")
    accuracies, root_client_accuracies = server.train(
        clients, data_loader_manager.get_root_loader(), #test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=True,
        root_client=root_client,
        fedprox=args.fedprox
    )
else:
    print("FedAvg Enabled!")
    accuracies = server.train(
        clients,data_loader_manager.get_root_loader(), # test_loader,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=False,
        root_client=None,
        fedprox=args.fedprox
    )
    root_client_accuracies = None

print("Global Model Accuracies across rounds:", accuracies)

# Uncomment to save matrices if needed for analysis
# save_matrices(A, B, C, args.attack_type, args.num_clients, args.num_malicious)
