import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from p1 import GCNEncoder, ContractFeatureExtractor  # Import necessary classes from p1.py

class P1ModelTrainer:
    def __init__(self, cache_folder, num_epochs=10, learning_rate=0.01, batch_size=32):
        self.cache_folder = cache_folder
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_extractor = ContractFeatureExtractor(cache_folder)
        self.model = None  # Will be initialized in the train method

    def train(self):
        """Train the GCN model defined in p1.py."""
        contracts = self.feature_extractor.process_contracts()  # Load contracts
        train_data, val_data = self.split_data(contracts)

        # Create DataLoader for training and validation
        train_loader = self.create_data_loader(train_data)
        val_loader = self.create_data_loader(val_data)

        # Initialize the model
        input_dim = 100 + 8  # Assuming 100 semantic + 8 syntactic features
        self.model = GCNEncoder(input_dim=input_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()  # Adjust based on your task

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for data in train_loader:
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.edge_attr)  # Forward pass
                loss = criterion(out, data.x)  # Assuming reconstruction loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(train_loader)}')

        # Save the trained model
        torch.save(self.model.state_dict(), "p1_model.pth")
        print("P1 model training completed and saved.")

        # Test the model after training
        self.test(val_loader)

    def test(self, val_loader):
        """Test the trained model on the validation dataset."""
        self.model.eval()
        total_loss = 0
        criterion = torch.nn.MSELoss()  # Adjust based on your task

        with torch.no_grad():
            for data in val_loader:
                out = self.model(data.x, data.edge_index, data.edge_attr)  # Forward pass
                loss = criterion(out, data.x)  # Assuming reconstruction loss
                total_loss += loss.item()

        print(f'Validation Loss: {total_loss / len(val_loader)}')

    def split_data(self, contracts):
        """Split the contracts into training and validation sets."""
        filenames = list(contracts.keys())
        train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)
        train_data = {file: contracts[file] for file in train_files}
        val_data = {file: contracts[file] for file in val_files}
        return train_data, val_data

    def create_data_loader(self, data):
        """Convert the data into a DataLoader format."""
        data_list = []
        for filename, (semantic, syntactic) in data.items():
            features = np.concatenate((semantic, syntactic))
            # Create a Data object for PyTorch Geometric
            data_list.append(Data(x=torch.tensor(features, dtype=torch.float32).unsqueeze(0), 
                                  edge_index=torch.empty((2, 0), dtype=torch.long), 
                                  edge_attr=torch.empty((0,), dtype=torch.float32)))
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

# Example usage in your training script
if __name__ == "__main__":
    cache_directory = "cache"  # Adjust this to your cache directory
    trainer = P1ModelTrainer(cache_directory, num_epochs=10, learning_rate=0.01)
    trainer.train()