import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import os
from datetime import datetime

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



# Plotting and saving the graph
import matplotlib.pyplot as plt
import os
from datetime import datetime

import matplotlib.pyplot as plt
import os
from datetime import datetime

def PlotResult(accuracies, root_accuracies=None, fltrust_enabled=False, num_clients=15, num_rounds=20, num_malicious=10):
    plt.figure(figsize=(10, 6))

    # Plot global model accuracies
    plt.plot(range(1, len(accuracies) + 1), accuracies, label=f'{"FLTrust" if fltrust_enabled else "FedAvg"} Accuracies', marker='o')

    # Plot root client accuracies if applicable
    if fltrust_enabled and root_accuracies:
        plt.plot(range(1, len(root_accuracies) + 1), root_accuracies, label='Root Client Accuracies', marker='x')

    # Add labels, title, and text info
    plt.xlabel('Round'), plt.ylabel('Accuracy (%)')
    plt.title(f'Global Model Accuracy {"with FLTrust" if fltrust_enabled else "with FedAvg"}')
    plt.text(0.05, 0.95, f'Clients: {num_clients}\nRounds: {num_rounds}\nMalicious: {num_malicious}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))
    
    plt.legend()

    # Save the plot
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_type = 'FLTrust' if fltrust_enabled else 'FedAvg'
    file_name = f"results/Run_{run_type}_Clients_{num_clients}_Rounds_{num_rounds}_Malicious_{num_malicious}_{timestamp}.png"
    plt.savefig(file_name)
    print(f"Graph saved as {file_name}")


# Call the save function after training


# Function to save histogram of model weights for each communication round
def save_histogram_of_weights(model_state_dict, round_num, clientid, folder='HistoRounds'):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Flatten all model weights into a single list
    all_weights = np.concatenate([v.cpu().numpy().flatten() for v in model_state_dict.values()])
    print(all_weights)
    # Dynamically calculate the number of bins based on the range of weight values
    min_weight, max_weight = np.min(all_weights), np.max(all_weights)
    # bins = np.linspace(min_weight, max_weight, num=50)  # Create 50 bins between min and max values
    bins = np.linspace(-0.1, 0.1, num=50)
    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f'Histogram of Model Weights - Round {round_num + 1}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    # Save the figure
    hist_path = os.path.join(folder, f'weights_histogram_round_{round_num + 1}_Client_{clientid+1}.png')
    plt.savefig(hist_path)
    plt.close()