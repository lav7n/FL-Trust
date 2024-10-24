import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import torchvision
from torchvision import transforms
import random

import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np

import os
import random
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class DriveDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, malicious_clients=None, client_id=None, attack_type=None, noise_stddev=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.malicious_clients = malicious_clients  # List of malicious clients (None for root/test datasets)
        self.client_id = client_id  # ID of this client (None for root/test datasets)
        self.attack_type = attack_type  # Type of attack (e.g., gaussian)
        self.noise_stddev = noise_stddev  # Standard deviation of noise

        self.image_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load the image and mask
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_name = self.image_list[idx].replace('_training.tif', '_manual1.gif')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("L")  # Convert to grayscale
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Apply Gaussian noise attack if this client is malicious (only for client datasets)
        if self.client_id is not None and self.client_id in self.malicious_clients and self.attack_type == 'gaussian':
            noise = torch.randn(image.size()) * self.noise_stddev / 255.0
            image = torch.clamp(image + noise, 0, 1)

        mask = (mask > 0).float()  # Binarize the mask

        return image, mask


class DataLoaderManager:
    def __init__(self, batch_size, num_clients, root_dataset_fraction, distribution='iid', num_malicious=0, attack_type=None, noise_stddev=256):
        self.batch_size = batch_size
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.attack_type = attack_type
        self.noise_stddev = noise_stddev
        self.distribution = distribution

        # Define image and mask transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Load DRIVE dataset
        self.train_set = DriveDataset(
            image_dir='/kaggle/input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/images',
            mask_dir='/kaggle/input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/1st_manual',
            transform=self.transform
        )

        self.test_set = DriveDataset(
            image_dir='/kaggle/input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/test/images',
            mask_dir='/kaggle/input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/test/1st_manual',
            transform=self.transform
        )

        # Select root dataset size and indices (client_id is None for root dataset)
        self.root_size = max(1, int(len(self.train_set) * root_dataset_fraction))
        self.root_indices = torch.randperm(len(self.train_set))[:self.root_size]
        self.root_dataset = Subset(self.train_set, self.root_indices)

        self.class_counts = torch.zeros(self.num_clients, 2)  # For segmentation (binary masks: 2 classes)
        self.root_class_counts = torch.zeros(1, 2)  # Root dataset class distribution

        self.malicious_clients = list(range(self.num_malicious))  # First 'num_malicious' clients are malicious

        self.CountClasses(self.root_indices, is_root=True)

        # Apply IID/Non-IID distribution
        if self.distribution == 'iid':
            self.IID()
        else:
            self.NonIID()

    def IID(self):
        # Distribute the remaining dataset equally across clients
        self.remaining_indices = list(set(range(len(self.train_set))) - set(self.root_indices))
        self.remaining_dataset = Subset(self.train_set, self.remaining_indices)
        
        client_size = max(1, len(self.remaining_dataset) // self.num_clients)
        self.client_datasets = []

        for i in range(self.num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size
            if start_idx < len(self.remaining_dataset):
                client_indices = range(start_idx, min(end_idx, len(self.remaining_dataset)))
            else:
                # Repeat indices if necessary to avoid empty datasets for clients
                client_indices = range(0, client_size)

            # Pass client_id and malicious_clients only for client datasets
            client_dataset = DriveDataset(
                image_dir='/kaggle/input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/images',
                mask_dir='/kaggle/input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/1st_manual',
                transform=self.transform,
                malicious_clients=self.malicious_clients,
                client_id=i,
                attack_type=self.attack_type,
                noise_stddev=self.noise_stddev
            )
            self.client_datasets.append(client_dataset)
            self.CountClasses(client_indices, i)

        self.DistributionMatrix()

    def NonIID(self):
        np.random.seed(42)
        random.seed(42)
        total_samples = len(self.train_set)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        self.client_datasets = [[] for _ in range(self.num_clients)]

        for i in range(self.num_clients):
            num_samples = random.randint(max(1, total_samples // (self.num_clients * 2)), total_samples // self.num_clients)
            selected_samples = np.random.choice(indices, size=num_samples, replace=False)
            self.client_datasets[i] = Subset(self.train_set, selected_samples)
            self.CountClasses(selected_samples, i)

        self.DistributionMatrix()

    def CountClasses(self, indices, client_id=None, is_root=False):
        # Count class distribution: mask has two values (binary: 0 and 1)
        # Add safeguard to handle small datasets or empty client datasets
        if len(indices) == 0:
            return
        
        masks = torch.stack([self.train_set[i][1] for i in indices])
        class_distribution = torch.bincount(masks.view(-1).int(), minlength=2).float()

        if is_root:
            self.root_class_counts[0] = class_distribution
        else:
            self.class_counts[client_id] = class_distribution

    def DistributionMatrix(self):
        print("Clients' distribution matrix (clients x classes):")
        print(self.class_counts)
        print("Root dataset distribution matrix (root x classes):")
        print(self.root_class_counts)

    def apply_attacks(self):
        # Label flipping attack: set all labels in the mask to 1
        if self.attack_type == 'label_flipping':
            print(f"Applying label flipping attack to {self.num_malicious} clients.")
            for i in range(self.num_malicious):
                for idx in self.client_datasets[i].indices:
                    image, mask = self.train_set[idx]
                    mask.fill_(1)  # Set all mask values to 1 (label flipping)
                    self.train_set[idx] = (image, mask)

        # Gaussian attack: add noise to images
        elif self.attack_type == 'gaussian':
            print(f"Applying Gaussian noise attack (stddev: {self.noise_stddev}) to {self.num_malicious} clients.")
            for i in range(self.num_malicious):
                for idx in self.client_datasets[i].indices:
                    image, mask = self.train_set[idx]
                    noise = torch.randn(image.size()) * self.noise_stddev / 255.0
                    noisy_image = torch.clamp(image + noise, 0, 1)
                    self.train_set[idx] = (noisy_image, mask)

    def get_root_loader(self):
        return DataLoader(self.root_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def get_client_loaders(self):
        return [DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True) for client_dataset in self.client_datasets]


def save_matrices(A, B, C, attack_type, num_clients, num_malicious):
    """
    Save the matrices A, B, and C as images in the 'matrix_results' directory.

    Args:
        A (np.ndarray): Matrix A representing the root client on clients' data.
        B (np.ndarray): Matrix B representing the clients on root client's data.
        C (np.ndarray): Matrix C representing cosine similarity between clients and root client.
        attack_type (str): The type of attack used during training.
        num_clients (int): Total number of clients.
        num_malicious (int): Number of malicious clients.
    """

    # Create the directory if it doesn't exist
    results_dir = f'matrix_results/{attack_type}_{num_clients}clients_{num_malicious}malicious'
    os.makedirs(results_dir, exist_ok=True)

    # Save Matrix A
    plt.figure(figsize=(6, 6))
    im_A = plt.imshow(A, cmap='magma', interpolation='none')
    plt.title('Root model on Clients data (Matrix A)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', color='white', fontsize=4)
    plt.colorbar(im_A)
    plt.savefig(os.path.join(results_dir, 'matrix_A.png'))
    plt.close()

    # Save Matrix B
    plt.figure(figsize=(6, 6))
    im_B = plt.imshow(B, cmap='magma', interpolation='none')
    plt.title('Client Models on Root Data (Matrix B)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            plt.text(j, i, f'{B[i, j]:.1f}', ha='center', va='center', color='white', fontsize=4)
    plt.colorbar(im_B)
    plt.savefig(os.path.join(results_dir, 'matrix_B.png'))
    plt.close()

    # Prepare and save Matrix C (Cosine similarity matrix)
    C = np.clip(C, -1, 1)  # Ensure values are within [-1, 1]
    C = np.round(C, 2)     # Round values to 2 decimal points

    plt.figure(figsize=(6, 6))
    im_C = plt.imshow(C, cmap='magma', interpolation='none')
    plt.title('Cosine Similarity Matrix (Matrix C)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', color='white', fontsize=4)
    plt.colorbar(im_C)
    plt.savefig(os.path.join(results_dir, 'matrix_C.png'))
    plt.close()

    print(f"Matrices saved to {results_dir}")