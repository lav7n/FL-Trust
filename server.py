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
    def __init__(self, model, criterion, num_clients, alpha=1, mu=0.0, print_metrics=False):
        self.model = model.to(device)
        self.criterion = criterion
        self.num_clients = num_clients
        self.alpha = alpha  # Trust-weighting factor for FLTrust
        self.mu = mu  # Proximal term weight for FedProx
        self.print_metrics = print_metrics

    def cosine(self, w1, w2):
        dot_product = np.dot(w1, w2)
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        return dot_product / (norm1 * norm2)

    def FLTrust(self, root_client, client_models):
        root_delta = {k: root_client.get_model_weights()[k] - self.model.state_dict()[k] for k in self.model.state_dict().keys()}
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

            cosine_sim = self.cosine(root_weights_flattened, client_weights_flattened)
            cosine_sim = max(cosine_sim, 0)

            norm_factor = np.linalg.norm(root_weights_flattened) / (np.linalg.norm(client_weights_flattened) + 1e-8)
            normalized_trust_score = cosine_sim * norm_factor

            trust_scores.append(normalized_trust_score)
            total_trust_score += normalized_trust_score
            deltas.append(client_delta)

            if self.print_metrics:
                print(f"Client {client_id + 1} - Trust Score: {cosine_sim:.4f}")

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

    def train(self, clients, test_loader, num_rounds=5, num_epochs=1, FLTrust=False, root_client=None, fedprox=False):
        accuracies = []
        root_client_accuracies = []

        for rnd in tqdm(range(num_rounds)):
            tqdm.write(f"\n--- Round {rnd + 1}/{num_rounds} ---")

            if FLTrust:
                root_client.update_model_weights(self.model.state_dict())
                root_client.train()

            client_models = []
            client_accuracies = []
            for client in clients:
                client.update_model_weights(self.model.state_dict())
                client.train(fedprox=fedprox, mu=self.mu)
                client_models.append(client.get_model_weights())

                if self.print_metrics:
                    client_accuracy = self.test_client_locally(client, test_loader)
                    print(f"Client {client} Accuracy on Test: {client_accuracy:.2f}%")
                    client_accuracies.append(client_accuracy)

            if FLTrust and root_client:
                self.FLTrust(root_client, client_models)
            else:
                self.FedAvg(client_models)

            global_accuracy = self.test_global(test_loader)
            accuracies.append(global_accuracy)
            tqdm.write(f"Global Model Accuracy after Round {rnd + 1}: {global_accuracy:.2f}%")

            if FLTrust and root_client:
                root_client_accuracy = self.test_client_locally(root_client, test_loader)
                root_client_accuracies.append(root_client_accuracy)
                tqdm.write(f"Root Client Accuracy after Round {rnd + 1}: {root_client_accuracy:.2f}%")

        return accuracies, root_client_accuracies

    def FedAvg(self, client_models):
        avg_weights = client_models[0]
        for key in avg_weights.keys():
            avg_weights[key] = torch.stack([client_models[i][key].float().to(device) for i in range(len(client_models))], dim=0).mean(dim=0)
        self.model.load_state_dict(avg_weights)

    def test_global(self, data_loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(data_loader.dataset)

    def test_client_locally(self, client, data_loader):
        client.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = client.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(data_loader.dataset)

