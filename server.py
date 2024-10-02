import numpy as np
import torch
import copy

# Detect if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Server:
    def __init__(self, model, criterion, num_clients, alpha=1):
        # Move the model to the appropriate device (CPU or GPU)
        self.model = model.to(device)
        self.criterion = criterion
        self.num_clients = num_clients
        self.alpha = alpha  # Trust-weighting factor for FLTrust

    def Cosine(self, w1, w2):
        dot_product = np.dot(w1, w2)
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        return dot_product / (norm1 * norm2)

    def FLTrust(self, root_client, client_models):
        # Train root client and get its deltas
        root_client.update_model_weights(self.model.state_dict())  # Set global weights before Training
        root_client.train()
        root_delta = {k: root_client.get_model_weights()[k] - self.model.state_dict()[k] 
                      for k in self.model.state_dict().keys()}

        # Flatten root client's delta into a 1D array
        root_weights_flattened = np.concatenate([param.cpu().numpy().ravel() for param in root_delta.values()])

        total_trust_score = 0
        trust_scores = []
        deltas = []

        # Calculate trust scores and deltas for each client
        for client_id, client_model in enumerate(client_models):
            # Compute client delta
            client_delta = {k: client_model[k] - self.model.state_dict()[k] for k in self.model.state_dict().keys()}
            client_weights_flattened = np.concatenate([param.cpu().numpy().ravel() for param in client_delta.values()])

            # Calculate cosine similarity as a trust score
            cosine_sim = self.Cosine(root_weights_flattened, client_weights_flattened)
            trust_score = max(cosine_sim, 0)  # Trust score must be non-negative

            # Normalize the trust score using norms
            norm_factor = np.linalg.norm(root_weights_flattened) / np.linalg.norm(client_weights_flattened)
            normalized_trust_score = trust_score * norm_factor

            # Store values
            trust_scores.append(normalized_trust_score)
            total_trust_score += normalized_trust_score
            deltas.append(client_delta)

            # Print Trust Score and Cosine Similarity for each client
            print(f"Client {client_id + 1} - Trust Score: {trust_score:.4f}")

        # Aggregate updates with FLTrust weights
        delta_weight = {k: trust_scores[0] * deltas[0][k] for k in deltas[0].keys()}
        for i in range(1, len(deltas)):
            for k in delta_weight:
                delta_weight[k] += trust_scores[i] * deltas[i][k]

        # Normalize by total trust score and update the global model
        for k in delta_weight:
            delta_weight[k] /= total_trust_score
            self.model.state_dict()[k] += self.alpha * delta_weight[k]

        # Load the updated weights into the global model
        self.model.load_state_dict(self.model.state_dict())

    def Train(self, clients, test_loader, num_rounds=5, num_epochs=1, FLTrust=False, root_client=None, test_global_model=True, root_client_only=False):
        accuracies = []
        root_client_accuracies = []

        # Initialize matrices to store accuracies
        root_on_client_matrix = np.zeros((num_rounds, self.num_clients))
        client_on_root_matrix = np.zeros((num_rounds, self.num_clients))

        # Test initial global model  
        if test_global_model:
            initial_accuracy = self.test_global(test_loader)
            accuracies.append(initial_accuracy)
            print(f'Initial Global Model Accuracy: {initial_accuracy:.2f}%')

        # Test root client before Training
        if FLTrust and root_client and test_global_model:
            root_client_initial_accuracy = self.test_client_locally(root_client, root_client.train_loader)
            root_client_accuracies.append(root_client_initial_accuracy)
            print(f'Initial Root Client Accuracy: {root_client_initial_accuracy:.2f}%')

        # Training rounds
        for rnd in range(num_rounds):
            print(f"\n--- Round {rnd + 1}/{num_rounds} ---")

            # ROOT CLIENT ONLY
            if root_client_only and root_client:
                print("Training only the root client...")
                root_client.update_model_weights(self.model.state_dict())  # Set global weights before Training
                root_client.train()
                root_delta = {k: root_client.get_model_weights()[k] - self.model.state_dict()[k] 
                            for k in self.model.state_dict().keys()}

                # Update global model with root client's updates
                for k in root_delta:
                    self.model.state_dict()[k] += self.alpha * root_delta[k]

                # Test root client and global model
                root_client_accuracy = self.test_client_locally(root_client, root_client.train_loader)
                root_client_accuracies.append(root_client_accuracy)
                print(f'Root Client Accuracy after Round {rnd + 1}: {root_client_accuracy:.2f}%')

                if test_global_model:
                    global_accuracy = self.test_global(test_loader)
                    accuracies.append(global_accuracy)
                    print(f'Global Model Accuracy after Round {rnd + 1}: {global_accuracy:.2f}%')

                continue

            # TRAIN ALL CLIENTS 
            client_models = []
            for client_id, client in enumerate(clients):
                client.update_model_weights(self.model.state_dict())
                client.train(num_epochs=num_epochs)
                client_models.append(client.get_model_weights())

                # Test client accuracy after local Training
                client_accuracy = self.test_client_locally(client, client.train_loader)
                print(f"Client {client_id + 1} - Accuracy: {client_accuracy:.2f}%")
                client_accuracy2 = self.test_client_locally(client, test_loader)
                print(f"Client {client_id + 1} - Accuracy on Test: {client_accuracy2:.2f}%")

                # Test the root client on the current client's data
                root_on_client_accuracy = self.test_client_locally(root_client, client.train_loader)
                print(f'\n Root Client Accuracy on Client {client_id + 1}: {root_on_client_accuracy:.2f}%')
                root_on_client_matrix[rnd, client_id] = root_on_client_accuracy

                # Test the current client on the root client's data
                client_on_root_accuracy = self.test_client_locally(client, root_client.train_loader)
                print(f'Client {client_id + 1} Accuracy on Root Client: {client_on_root_accuracy:.2f}% \n\n')
                client_on_root_matrix[rnd, client_id] = client_on_root_accuracy

            # FLTRUST
            if FLTrust and root_client:
                self.FLTrust(root_client, client_models)
                root_client_accuracy = self.test_client_locally(root_client, root_client.train_loader)
                root_client_accuracies.append(root_client_accuracy)
                print(f'Root Client Accuracy after Round {rnd + 1}: {root_client_accuracy:.2f}%')

                root_client_accuracy2 = self.test_client_locally(root_client, test_loader)
                print(f'Root Client Accuracy on test set after Round {rnd + 1}: {root_client_accuracy2:.2f}%')
            else:
                self.FedAvg(client_models)

            # Test global model after aggregation
            if test_global_model:
                global_accuracy = self.test_global(test_loader)
                accuracies.append(global_accuracy)
                print(f'Global Model Accuracy after Round {rnd + 1}: {global_accuracy:.2f}%')

        return accuracies, root_client_accuracies, root_on_client_matrix, client_on_root_matrix


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
                data = data.view(data.size(0), 1, 28, 28)
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
                data = data.view(data.size(0), 1, 28, 28)
                output = client.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(data_loader.dataset)
