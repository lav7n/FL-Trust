import numpy as np
import torch
import copy
import shutil
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Server:
    def __init__(self, model, criterion, num_clients, alpha=1, LRPoison=False, printmetrics=False):
        # Move the model to the appropriate device (CPU or GPU)
        self.model = model.to(device)
        self.criterion = criterion
        self.num_clients = num_clients
        self.alpha = alpha  # Trust-weighting factor for FLTrust
        self.LRPoison = LRPoison
        self.printmetrics = printmetrics

    def Cosine(self, w1, w2):
        dot_product = np.dot(w1, w2)
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        return dot_product / (norm1 * norm2)


    def FLTrust(self, root_client, client_models):
        root_delta = {k: root_client.get_model_weights()[k] - self.model.state_dict()[k] 
                    for k in self.model.state_dict().keys()}
        root_weights_flattened = np.concatenate([param.cpu().numpy().ravel() for param in root_delta.values()])

        total_trust_score = 0
        trust_scores = []
        deltas = []

        for client_id, client_model in enumerate(client_models):
            client_delta = {
                k: torch.where(
                    torch.isnan(client_model[k]) | 
                    torch.isnan(self.model.state_dict()[k]) | 
                    torch.isnan(client_model[k] - self.model.state_dict()[k]), 
                    torch.zeros_like(client_model[k]), 
                    client_model[k] - self.model.state_dict()[k]
                )
                for k in self.model.state_dict().keys()
            }
            
            client_weights_flattened = np.concatenate([param.cpu().numpy().ravel() for param in client_delta.values()])

            cosine_sim = self.Cosine(root_weights_flattened, client_weights_flattened)

            if np.isnan(cosine_sim):
                cosine_sim = 0
            
            trust_score = max(cosine_sim, 0)

            norm_factor = np.linalg.norm(root_weights_flattened) / (np.linalg.norm(client_weights_flattened) + 1e-8)
            normalized_trust_score = trust_score * norm_factor
            
            trust_scores.append(normalized_trust_score)
            total_trust_score += normalized_trust_score
            deltas.append(client_delta)

            if self.printmetrics:
                print(f"Client {client_id + 1} - Trust Score: {trust_score:.4f}")

        delta_weight = {k: trust_scores[0] * deltas[0][k] for k in deltas[0].keys()}
        for i in range(1, len(deltas)):
            for k in delta_weight:
                delta_weight[k] += trust_scores[i] * deltas[i][k]

        if total_trust_score > 0:
            for k in delta_weight:
                delta_weight[k] /= total_trust_score
                if isinstance(self.model.state_dict()[k], torch.Tensor):
                    self.model.state_dict()[k] += self.alpha * delta_weight[k].type_as(self.model.state_dict()[k])

        self.model.load_state_dict(self.model.state_dict())

    def Train(self, clients, test_loader, num_rounds=5, num_epochs=1, FLTrust=False, root_client=None, test_global_model=True, root_client_only=False):
        accuracies = []
        root_client_accuracies = []

        root_on_client_matrix = np.zeros((num_rounds, self.num_clients))
        client_on_root_matrix = np.zeros((num_rounds, self.num_clients))
        similarity_matrix = np.zeros((num_rounds, self.num_clients))

        for rnd in tqdm(range(num_rounds)):
            tqdm.write(f"\n--- Round {rnd + 1}/{num_rounds} ---")

            # TRAIN ALL CLIENTS AND ROOT
            root_client.update_model_weights(self.model.state_dict())  # Set global weights before Training
            root_client.train() 

            client_models = []
            for client_id in range(len(clients)):
                client = clients[client_id]
                client.update_model_weights(self.model.state_dict())
                client.train(num_epochs=num_epochs)
                client_models.append(client.get_model_weights())
                # save_histogram_of_weights(client.get_model_weights(), rnd, client_id)

                # Test client accuracy after local training
                # client_accuracy = self.test_client_locally(client, client.train_loader)
                # print(f"Client {client_id + 1} - Accuracy: {client_accuracy:.2f}%")
                # client_accuracy2 = self.test_client_locally(client, test_loader)
                # print(f"Client {client_id + 1} - Accuracy on Test: {client_accuracy2:.2f}%")

                if self.printmetrics:
                    client_accuracy = self.test_client_locally(client, client.train_loader)
                    print(f"Client {client_id + 1} - Accuracy: {client_accuracy:.2f}%")

            if FLTrust and root_client: #Testing Root Client
                self.FLTrust(root_client, client_models)
            else:
                self.FedAvg(client_models)

            # Test global model accuracy
            global_accuracy = self.test_global(test_loader)
            accuracies.append(global_accuracy)
            tqdm.write(f"Global Model Accuracy after Round {rnd + 1}: {global_accuracy:.2f}%")

        return accuracies, root_client_accuracies, root_on_client_matrix, client_on_root_matrix, similarity_matrix


    def FedAvg(self, client_models):
        """Average the model weights from clients for FedAvg aggregation."""
        # Assume client_models already contains the state dictionaries (OrderedDicts)
        avg_weights = client_models[0]  # Directly use the first client's state_dict

        # Stack all model parameters along the first dimension and compute the mean
        for key in avg_weights.keys():
            avg_weights[key] = torch.stack([client_models[i][key].float().to(device) for i in range(len(client_models))], dim=0).mean(dim=0)
        
        # Load the averaged weights into the model
        self.model.load_state_dict(avg_weights)

    def test_global(self, data_loader):
        """Evaluate global model performance on test data."""
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), 3, 32, 32) 
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(data_loader.dataset)

    def test_client_locally(self, client, data_loader):
        """Evaluate a single client's model on their own data."""
        client.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), 3, 32, 32) 
                output = client.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(data_loader.dataset)
