import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image

class DataLoaderManager:
    def __init__(self, 
                 image_dir="/kaggle/input/2dbrats/Training/Training/Images", 
                 mask_dir="/kaggle/input/2dbrats/Training/Training/Masks", 
                 batch_size=8, 
                 num_clients=2, 
                 root_dataset_fraction=0.1, 
                 distribution='iid', 
                 num_malicious=1, 
                 attack_type='label_flipping', 
                 noise_stddev=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
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

        # Load all file paths
        self.image_list = sorted(os.listdir(self.image_dir))

        # Create the full dataset and split into train and test
        train_size = int(0.8 * len(self.image_list))
        test_size = len(self.image_list) - train_size
        self.train_indices, self.test_indices = random_split(range(len(self.image_list)), [train_size, test_size])

        # Root dataset selection
        self.root_size = max(1, int(len(self.train_indices) * root_dataset_fraction))
        self.root_indices = torch.randperm(len(self.train_indices))[:self.root_size]
        self.root_dataset = Subset(self, self.root_indices)

        # Set up client datasets based on distribution
        if self.distribution == 'iid':
            self.IID()
        else:
            self.NonIID()

        # Identify malicious clients
        self.malicious_clients = list(range(self.num_malicious))
        print(f"Malicious clients (indices): {self.malicious_clients}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx, client_id=None):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = np.array(mask)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8) 
        mask = mask.astype(np.float32)
        
        if client_id is not None and client_id in self.malicious_clients:
            mask = self.apply_attack(mask, client_id)

        return image, mask


    def apply_attack(self, mask, client_id):
        # Ensure mask is a torch tensor
        mask = torch.tensor(mask, dtype=torch.float32) if not isinstance(mask, torch.Tensor) else mask
        
        if client_id in self.malicious_clients:
            if self.attack_type == 'gaussian':
                noise = torch.randn(mask.shape) * (self.noise_stddev /255)
                mask = torch.clamp(mask + noise, 0, 1)
                # print(f"Client {client_id} - Gaussian noise attack applied")
            elif self.attack_type == 'label_flipping':
                mask = 1 - mask
                # print(f"Client {client_id} - Label flipping attack applied")
        return mask

    def IID(self):
        remaining_indices = list(set(range(len(self.train_indices))) - set(self.root_indices))
        client_size = max(1, len(remaining_indices) // self.num_clients)

        self.client_datasets = [
            Subset(self, range(i * client_size, min((i + 1) * client_size, len(remaining_indices))))
            for i in range(self.num_clients)
        ]
        self.DistributionMatrix()

    def NonIID(self):
        np.random.seed(42)
        random.seed(42)
        total_samples = len(self.train_indices)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        self.client_datasets = []
        for i in range(self.num_clients):
            num_samples = random.randint(max(1, total_samples // (self.num_clients * 2)), total_samples // self.num_clients)
            selected_samples = np.random.choice(indices, size=num_samples, replace=False)
            self.client_datasets.append(Subset(self, selected_samples))
        self.DistributionMatrix()

    def DistributionMatrix(self):
        print("Number of samples per client:")
        for i, client_dataset in enumerate(self.client_datasets):
            print(f"Client {i + 1}: {len(client_dataset)} samples")
        print(f"Root dataset: {len(self.root_dataset)} samples")

    def get_root_loader(self):
        return DataLoader(self.root_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        test_dataset = Subset(self, self.test_indices)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_client_loaders(self):
        loaders = []
        for i, client_dataset in enumerate(self.client_datasets):
            # Create a custom dataset wrapper to apply attack based on client_id
            client_dataset_with_attack = ClientDatasetWrapper(self, client_dataset, i)
            loaders.append(DataLoader(client_dataset_with_attack, batch_size=self.batch_size, shuffle=True))
        return loaders


class ClientDatasetWrapper(Dataset):
    def __init__(self, data_manager, subset, client_id):
        self.data_manager = data_manager
        self.subset = subset
        self.client_id = client_id

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Retrieve the global index for the subset index
        global_idx = self.subset.indices[idx]
        # Pass the client_id to apply the attack if the client is malicious
        return self.data_manager.__getitem__(global_idx, self.client_id)

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