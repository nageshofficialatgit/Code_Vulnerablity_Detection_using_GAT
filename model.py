import os
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sc import  process_sol_files
# Define GNN model
class SolidityGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SolidityGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Final output layer
        x = self.linear(x)
        return x

# Function to load data for one Solidity file
def load_solidity_data(sol_folder):
    process_sol_files(sol_folder)
    """Load data for one Solidity file."""
    graph_path = sol_folder / 'function_graph.gml'
    G = nx.read_gml(graph_path)
    
    # Load node features (embeddings)
    node_features = []
    node_mapping = {node: idx for idx, node in enumerate(G.nodes)}
    for node in G.nodes:
        embedding_path = sol_folder / f"{node}.npy"
        if embedding_path.exists():
            node_features.append(np.load(embedding_path))
        else:
            node_features.append(np.zeros(128))  # Default if embedding missing
    
    # Create edge index
    edge_index = []
    for src, dst in G.edges:
        edge_index.append([node_mapping[src], node_mapping[dst]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    
    # Convert to PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index
    )
    return data

def train_solidity_gnn(cache_dir, num_epochs=10, learning_rate=0.01):
    """Train the SolidityGNN model."""
    # Load all Solidity files
    data_list = []
    for sol_folder in Path(cache_dir).iterdir():
        if sol_folder.is_dir():
            print(f"Loading data from: {sol_folder}")
            data_list.append(load_solidity_data(sol_folder))

    # Create a DataLoader
    batch_size = 4  # Adjust batch size as needed
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with batch size: {batch_size}")

    # Initialize model
    in_channels = 128
    hidden_channels = 64
    out_channels = 64
    model = SolidityGNN(in_channels, hidden_channels, out_channels)

    # Generate embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Adjust based on your task

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()  # Zero the gradients
            out = model(data.x, data.edge_index, data.batch)  # Forward pass
            
            # Assuming you have target values for your embeddings
            target = data.y  # Replace with your actual target tensor
            loss = criterion(out, target)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            total_loss += loss.item()  # Accumulate loss

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), "solidity_gnn_model.pth")
    print("Model training completed and saved.")

def extract_embeddings(cache_dir):
    """Extract and save final embeddings from the SolidityGNN model."""
    # Load all Solidity files
    data_list = []
    folder_names = []  # To store folder names for embeddings
    for sol_folder in Path(cache_dir).iterdir():
        if sol_folder.is_dir():
            print(f"Loading data from: {sol_folder}")
            data = load_solidity_data(sol_folder)
            if data is not None:
                data_list.append(data)
                folder_names.append(sol_folder.name)  # Store the folder name

    # Check if data_list is empty
    if not data_list:
        print("No data loaded. Exiting.")
        return

    # Create a DataLoader
    batch_size = 4  # Adjust batch size as needed
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    print(f"DataLoader created with batch size: {batch_size}")

    # Initialize model
    in_channels = 128
    hidden_channels = 64
    out_channels = 64
    model = SolidityGNN(in_channels, hidden_channels, out_channels)

    # Load the model weights if you have a pre-trained model
    # model.load_state_dict(torch.load("solidity_gnn_model.pth"))  # Uncomment if you have a saved model

    # Generate embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Forward pass
            for i in range(out.size(0)):  # Iterate over the batch size
                embeddings[folder_names[i]] = out[i].cpu().numpy()  # Save embeddings per graph

    # Save embeddings to a file
    output_path = "solidity_embeddings.npy"
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

# Example usage in your training script
if __name__ == "__main__":
    cache_directory = "cache"  # Adjust this to your cache directory
    #train_solidity_gnn(cache_directory, num_epochs=10, learning_rate=0.01)
    extract_embeddings(cache_directory)
