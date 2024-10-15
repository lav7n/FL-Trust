import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class RootClientDataLoader:
    def __init__(self, batch_size=64):
        root_dir = 'dataset/root'  # Directory where root client's data is stored
        x_root, y_root = load_npy_data(root_dir)
        self.train_loader = DataLoader(TensorDataset(x_root, y_root), batch_size=batch_size, shuffle=True)

    def get_dataloader(self):
        """Returns the DataLoader for the root client."""
        return self.train_loader

class ClientDataLoader:
    def __init__(self, num_clients, num_malicious=0, batch_size=64, attack_type=None, noise_stddev=0):
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.batch_size = batch_size
        self.attack_type = attack_type
        self.noise_stddev = noise_stddev
        self.client_datasets = []
        self.malicious_clients = set(np.random.choice(range(num_clients), num_malicious, replace=False))
        self._load_client_data()

    def _load_client_data(self):
        for client_id in range(self.num_clients):
            client_dir = f'dataset/client_{client_id + 1}'
            x_data, y_data = load_npy_data(client_dir)
            
            # Apply attacks to malicious clients
            if client_id in self.malicious_clients:
                if self.attack_type == 'label_flipping':
                    y_data[:] = 9  # Flip all labels to class 9
                elif self.attack_type == 'gaussian_noise':
                    noise = torch.normal(mean=0, std=self.noise_stddev, size=x_data.size())
                    print(f"Adding Gaussian noise with stddev {self.noise_stddev}")
                    x_data = x_data + noise

            train_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=self.batch_size, shuffle=True)
            self.client_datasets.append(train_loader)

    def get_client_datasets(self):
        return self.client_datasets

    def get_malicious_clients(self):
        return list(self.malicious_clients)

class TestDataLoader:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.test_loader = None
        self._load_test_data()

    def _load_test_data(self):
        test_dir = 'dataset/test'
        x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(test_dir, 'y_test.npy'))

        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        self.test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        return self.test_loader


def load_npy_data(client_dir):
    x_data = np.load(os.path.join(client_dir, 'x_data.npy'))
    y_data = np.load(os.path.join(client_dir, 'y_data.npy'))
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long)


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