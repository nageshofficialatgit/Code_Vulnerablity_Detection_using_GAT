import os
import re
import logging
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from slither.slither import Slither
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_function_call(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Running function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# TextCNN Model
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Switch to (batch, channels, length) for Conv1d
        conv_outs = [self.relu(conv(x)).max(dim=2)[0] for conv in self.convs]  # Max-pooling
        return torch.cat(conv_outs, dim=1)

# Transformer Encoder Model with batch_first=True
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.transformer(x).mean(dim=1)  # Mean pooling over sequence

# Define GNN model
class SolidityGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SolidityGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

@log_function_call
def extract_functions_from_file(sol_file_path):
    logging.debug(f"Extracting functions from: {sol_file_path}")
    slither = Slither(sol_file_path)
    functions = {}
    with open(sol_file_path, 'r') as file:
        content = file.read()

    for contract in slither.contracts:
        for function in contract.functions:
            function_name = function.name
            start_byte = function.source_mapping.start
            length = function.source_mapping.length
            end_byte = start_byte + length
            function_content = content[start_byte:end_byte]
            functions[function_name] = function_content

    logging.info(f"Extracted {len(functions)} functions.")
    return functions

@log_function_call
def generate_embeddings(functions):
    logging.debug("Generating Word2Vec embeddings.")
    tokenized_functions = [re.findall(r'\w+', func) for func in functions.values()]
    model = Word2Vec(sentences=tokenized_functions, vector_size=100, window=5, min_count=1, workers=4)
    embeddings = {name: model.wv[re.findall(r'\w+', func)] for name, func in functions.items()}
    logging.info("Generated embeddings for all functions.")
    return embeddings

@log_function_call
def build_function_graph(functions):
    logging.debug("Building function graph.")
    graph = nx.DiGraph()
    function_names = list(functions.keys())
    for func_name in function_names:
        graph.add_node(func_name)
    for func_name, content in functions.items():
        for target_func in function_names:
            if target_func in content and target_func != func_name:
                graph.add_edge(func_name, target_func)
    logging.info(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

@log_function_call
def generate_feature_embeddings(embeddings, embedding_dim=100, num_filters=64, kernel_sizes=[3, 4, 5], nhead=4, num_layers=2):
    logging.debug("Generating feature embeddings using TextCNN and Transformer.")
    textcnn = TextCNN(embedding_dim=embedding_dim, num_filters=num_filters, kernel_sizes=kernel_sizes)
    transformer = TransformerEncoder(embedding_dim=num_filters * len(kernel_sizes), nhead=nhead, num_layers=num_layers)
    feature_embeddings = {}

    for func_name, func_embedding in embeddings.items():
        func_embedding_tensor = torch.tensor(func_embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        textcnn_out = textcnn(func_embedding_tensor)
        transformer_out = transformer(textcnn_out.unsqueeze(1))  # Transformer expects (batch, sequence, embedding) format
        feature_embeddings[func_name] = transformer_out.squeeze(0).detach().numpy()

    logging.info("Feature embeddings generated.")
    return feature_embeddings

@log_function_call
def process_sol_files(cache_folder):
    logging.info(f"Processing Solidity files in: {cache_folder}")
    for sol_file in os.listdir(cache_folder):
        if sol_file.endswith(".sol"):
            sol_path = os.path.join(cache_folder, sol_file)
            file_name = os.path.splitext(sol_file)[0]
            output_folder = os.path.join(cache_folder, file_name)
            os.makedirs(output_folder, exist_ok=True)

            # Step 1: Extract functions
            functions = extract_functions_from_file(sol_path)

            # Step 2: Generate Word2Vec embeddings
            embeddings = generate_embeddings(functions)

            # Step 3: Generate feature embeddings using TextCNN and Transformer
            feature_embeddings = generate_feature_embeddings(embeddings)
            for func_name, feature_emb in feature_embeddings.items():
                embedding_path = os.path.join(output_folder, f"{func_name}_feature_embedding.npy")
                np.save(embedding_path, feature_emb)

            # Step 4: Build and save function graph
            graph = build_function_graph(functions)
            graph_path = os.path.join(output_folder, "function_graph.gml")
            nx.write_gml(graph, graph_path)

            logging.info(f"Processed {sol_file}: Graph and feature embeddings saved in {output_folder}")

@log_function_call
def load_solidity_data(sol_folder):
    logging.debug(f"Loading data from: {sol_folder}")
    process_sol_files(sol_folder)
    graph_path = sol_folder / 'function_graph.gml'
    if not graph_path.exists():
        logging.error(f"Graph file not found: {graph_path}")
        return None

    try:
        G = nx.read_gml(graph_path)
    except Exception as e:
        logging.error(f"Failed to read graph file: {e}")
        return None

    node_features = []
    node_mapping = {node: idx for idx, node in enumerate(G.nodes)}
    for node in G.nodes:
        embedding_path = sol_folder / f"{node}.npy"
        if embedding_path.exists():
            node_features.append(np.load(embedding_path))
        else:
            logging.warning(f"Embedding file not found for node: {node}, using default.")
            node_features.append(np.zeros(128))  # Default if embedding missing
    
    if not node_features:
        logging.error("No node features loaded.")
        return None

    edge_index = []
    for src, dst in G.edges:
        edge_index.append([node_mapping[src], node_mapping[dst]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index
    )
    logging.info(f"Data loaded with {len(node_features)} nodes.")
    return data

@log_function_call
def train_solidity_gnn(cache_dir, num_epochs=10, learning_rate=0.01):
    logging.info(f"Training GNN model with data from: {cache_dir}")
    data_list = []
    for sol_folder in Path(cache_dir).iterdir():
        if sol_folder.is_dir():
            data = load_solidity_data(sol_folder)
            if data is not None:
                data_list.append(data)

    if not data_list:
        logging.error("No data loaded. Exiting.")
        return

    batch_size = 4
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    logging.debug(f"DataLoader created with batch size: {batch_size}")

    in_channels = 128
    hidden_channels = 64
    out_channels = 64
    model = SolidityGNN(in_channels, hidden_channels, out_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            
            target = data.y  # Replace with your actual target tensor
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

    torch.save(model.state_dict(), "solidity_gnn_model.pth")
    logging.info("Model training completed and saved.")

@log_function_call
def extract_embeddings(cache_dir):
    logging.info(f"Extracting embeddings from data in: {cache_dir}")
    data_list = []
    folder_names = []
    for sol_folder in Path(cache_dir).iterdir():
        if sol_folder.is_dir():
            logging.debug(f"Processing folder: {sol_folder}")
            data = load_solidity_data(sol_folder)
            if data is not None:
                data_list.append(data)
                folder_names.append(sol_folder.name)

    if not data_list:
        logging.error("No data loaded. Exiting.")
        return

    batch_size = 4
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    logging.debug(f"DataLoader created with batch size: {batch_size}")

    in_channels = 128
    hidden_channels = 64
    out_channels = 64
    model = SolidityGNN(in_channels, hidden_channels, out_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            for i in range(out.size(0)):
                embeddings[folder_names[i]] = out[i].cpu().numpy()

    output_path = "solidity_embeddings.npy"
    np.save(output_path, embeddings)
    logging.info(f"Embeddings saved to {output_path}")

# Example usage in your training script
if __name__ == "__main__":
    cache_directory = r"D:\acadmics\sem 5\Innovation paper\IJCAI IMPLE\seems\seems\cache"
    # Ensure that the data is processed before extracting embeddings
    process_sol_files(cache_directory)
    extract_embeddings(cache_directory)