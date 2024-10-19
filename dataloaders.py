import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import torchvision
from torchvision import transforms

class DataLoaderManager:
    def __init__(self, batch_size, num_clients, root_dataset_fraction, distribution='iid'):
        self.batch_size = batch_size
        self.num_clients = num_clients
        self.distribution = distribution
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

        self.root_size = int(len(self.train_set) * root_dataset_fraction)
        self.root_indices = torch.randperm(len(self.train_set))[:self.root_size]
        self.root_dataset = Subset(self.train_set, self.root_indices)

        self.class_counts = torch.zeros(self.num_clients, 10)  # Clients' class distribution matrix
        self.root_class_counts = torch.zeros(1, 10)  # Root dataset class distribution matrix

        self.CountClasses(self.root_indices, is_root=True)

        if self.distribution == 'iid':
            self.IID()
        else:
            self.NonIID()

    def IID(self):
        self.remaining_indices = list(set(range(len(self.train_set))) - set(self.root_indices))
        self.remaining_dataset = Subset(self.train_set, self.remaining_indices)
        client_size = len(self.remaining_dataset) // self.num_clients
        self.client_datasets = []
        
        for i in range(self.num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size
            client_indices = range(start_idx, end_idx)
            self.client_datasets.append(Subset(self.remaining_dataset, client_indices))
            self.CountClasses(client_indices, i)
        
        self.DistributionMatrix()

    def NonIID(self):
        targets = torch.tensor(self.train_set.targets)
        indices_by_class = [torch.where(targets == i)[0] for i in range(10)]
        
        self.client_datasets = [[] for _ in range(self.num_clients)]
        
        for i, class_indices in enumerate(indices_by_class):
            # Make sure the split sizes are non-negative and smaller than the number of available samples
            split_sizes = torch.randint(low=0, high=len(class_indices) // self.num_clients + 1, size=(self.num_clients,))
            split_sizes[-1] = len(class_indices) - split_sizes[:-1].sum()  # Adjust the last size to cover the remaining samples
            
            # Ensure the last split is non-negative by re-adjusting if necessary
            if split_sizes[-1] < 0:
                split_sizes[-1] = 0
            
            # Now perform the split
            splits = torch.split(class_indices, split_sizes.tolist())
            
            for j, split in enumerate(splits):
                self.client_datasets[j].extend(split.tolist())
                self.class_counts[j, i] += len(split)
        
        self.client_datasets = [Subset(self.train_set, indices) for indices in self.client_datasets]
        self.DistributionMatrix()

    def CountClasses(self, indices, client_id=None, is_root=False):
        targets = torch.tensor([self.train_set.targets[i] for i in indices])
        class_distribution = torch.bincount(targets, minlength=10)
        
        if is_root:
            self.root_class_counts[0] = class_distribution
        else:
            self.class_counts[client_id] = class_distribution

    def DistributionMatrix(self):
        print("Clients' distribution matrix (clients x classes):")
        print(self.class_counts)
        print("Root dataset distribution matrix (root x classes):")
        print(self.root_class_counts)

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