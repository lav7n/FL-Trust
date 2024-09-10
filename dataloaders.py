import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import struct

def load_npy_data(client_dir):
    x_data = np.load(os.path.join(client_dir, 'x_data.npy'))
    y_data = np.load(os.path.join(client_dir, 'y_data.npy'))
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long)

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



def load_mnist_images(filename):
    """Load MNIST images from an idx3-ubyte file."""
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions of the images
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, 1, rows, cols)
        # Normalize the pixel values to [0, 1]
        return images / 255.0

def load_mnist_labels(filename):
    """Load MNIST labels from an idx1-ubyte file."""
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def load_test_data():
    """Load the MNIST test data from idx3-ubyte and idx1-ubyte files."""
    # Load images and labels from the files
    images = load_mnist_images('mnist/t10k-images.idx3-ubyte')
    labels = load_mnist_labels('mnist/t10k-labels.idx1-ubyte')
    
    # Convert them to PyTorch tensors
    x_test_tensor = torch.tensor(images, dtype=torch.float32)
    y_test_tensor = torch.tensor(labels, dtype=torch.long)

    # Create a TensorDataset and DataLoader
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return test_loader