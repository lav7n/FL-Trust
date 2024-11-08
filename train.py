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
parser.add_argument('--num_rounds', type=int, default=10, help='Number of training rounds')
parser.add_argument('--num_malicious', type=int, default=15, help='Number of malicious clients')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for each client')
parser.add_argument('--FLTrust', action='store_true', help='Use FLTrust or not')
parser.add_argument('--FedProx', action='store_true', help='Use FedProx or not')
parser.add_argument('--attack_type', type=str, default='gaussian', help='Type of attack to apply to malicious clients')
parser.add_argument('--noise_stddev', type=float, default=64, help='Standard deviation of noise for Gaussian noise attack')
parser.add_argument('--printmetrics', action='store_true', help='Print metrics or not')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for clients')
parser.add_argument('--distribution', type=str, default='non_iid', help='Data distribution among clients')
parser.add_argument('--img_dir', type=str, default='/kaggle/input/2dbrats/Brats2d_Processed_First5000/images', help='Path to the directory containing images')
parser.add_argument('--seg_dir', type=str, default='/kaggle/input/2dbrats/Brats2d_Processed_First5000/masks', help='Path to the directory containing segmentation masks')
args = parser.parse_args()

model = smp.Unet(
    encoder_name="mobilenet_v2",     # Use EfficientNet-B0 as the encoder
    encoder_weights="imagenet",         # Pretrained on ImageNet
    in_channels=1,                      # Grayscale images
    classes=1                           # Binary segmentation
).to(device)
criterion = nn.BCEWithLogitsLoss()

# Initialize data loaders with potential attacks on malicious clients
data_loader_manager = DataLoaderManager(
    image_dir=args.img_dir,         # Use img_dir argument
    mask_dir=args.seg_dir,          # Use seg_dir argument
    batch_size=4,                   # Adjust batch size for segmentation task
    num_clients=args.num_clients, 
    root_dataset_fraction=0.1, 
    distribution=args.distribution,
    num_malicious=args.num_malicious,
    attack_type=args.attack_type,
    noise_stddev=args.noise_stddev
)

# Get test loader and client loaders
client_loaders = data_loader_manager.get_client_loaders()
test_loader = data_loader_manager.get_test_loader()

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

root_client = Client(client_loader=data_loader_manager.get_root_loader(), num_epochs=args.num_epochs, lr=default_lr)
server = Server(model=model, criterion=criterion, num_clients=args.num_clients, alpha=1, print_metrics=args.printmetrics)

# FLTrust vs FedAvg training loop
print("FLTrust: ", args.FLTrust)
print("FedProx: ", args.FedProx)
if args.FLTrust:
    print("FLTrust Enabled!")
    accuracies, root_client_accuracies = server.train(
        clients, test_loader,  # Pass test_loader as the second positional argument
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=True,
        root_client=root_client,
        FedProx=args.FedProx
    )
else:
    print("FedAvg Enabled!")
    accuracies = server.train(
        clients, test_loader,  # Pass test_loader as the second positional argument
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        FLTrust=False,
        root_client=None,
        FedProx=args.FedProx
    )
    root_client_accuracies = None


print("Global Model Accuracies across rounds:", accuracies)

# Uncomment to save matrices if needed for analysis
# save_matrices(A, B, C, args.attack_type, args.num_clients, args.num_malicious)
