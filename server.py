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
        iou_scores = []
        client_iou_scores = []
        root_client_iou_scores = []

        for rnd in tqdm(range(num_rounds)):
            tqdm.write(f"\n--- Round {rnd + 1}/{num_rounds} ---")

            if FLTrust:
                root_client.update_model_weights(self.model.state_dict())
                root_client.train()

            client_models = []
            for client in clients:
                client.update_model_weights(self.model.state_dict())
                client.train(fedprox=fedprox, mu=self.mu)
                client_models.append(client.get_model_weights())

                if self.print_metrics:
                    client_iou = self.test_client_locally(client, test_loader)
                    print(f"Client IoU on Test: {client_iou:.2f}")
                    client_iou_scores.append(client_iou)

            if FLTrust and root_client:
                self.FLTrust(root_client, client_models)
            else:
                self.FedAvg(client_models)

            global_iou = self.test_global(test_loader)
            iou_scores.append(global_iou)
            tqdm.write(f"Global Model IoU after Round {rnd + 1}: {global_iou:.2f}")

            if FLTrust and root_client:
                root_client_iou = self.test_client_locally(root_client, test_loader)
                root_client_iou_scores.append(root_client_iou)
                tqdm.write(f"Root Client IoU after Round {rnd + 1}: {root_client_iou:.2f}")

        return iou_scores, root_client_iou_scores

    def FedAvg(self, client_models):
        avg_weights = client_models[0]
        for key in avg_weights.keys():
            avg_weights[key] = torch.stack([client_models[i][key].float().to(device) for i in range(len(client_models))], dim=0).mean(dim=0)
        self.model.load_state_dict(avg_weights)

    def test_global(self, data_loader):
        self.model.eval()
        iou_score = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                
                # For segmentation tasks, use sigmoid and threshold
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                # Compute Intersection over Union (IoU)
                intersection = (preds * target).sum((1, 2, 3))
                union = (preds + target).sum((1, 2, 3)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero

                iou_score += iou.mean().item()
                total += 1

        return iou_score / total  # Average IoU across the dataset

    def test_client_locally(self, client, data_loader):
        client.model.eval()
        iou_score = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputs = client.model(data)
                
                # For segmentation tasks, use sigmoid and threshold
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                # Compute IoU for client model
                intersection = (preds * target).sum((1, 2, 3))
                union = (preds + target).sum((1, 2, 3)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)

                iou_score += iou.mean().item()
                total += 1

        return iou_score / total  # Average IoU for this client
