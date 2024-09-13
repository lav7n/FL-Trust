import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import struct

class RootClientDataLoader:
    def __init__(self, batch_size=64):
        root_dir = 'dataset/root_client'  # Directory where root client's data is stored
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
            client_dir = f'dataset/clients/client_{client_id + 1}'
            x_data, y_data = load_npy_data(client_dir)
            
            # Apply attacks to malicious clients
            if client_id in self.malicious_clients:
                if self.attack_type == 'label_flipping':
                    y_data[:] = 9  # Flip all labels to class 9
                elif self.attack_type == 'gaussian_noise':
                    noise = torch.normal(mean=0, std=self.noise_stddev, size=x_data.size())
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