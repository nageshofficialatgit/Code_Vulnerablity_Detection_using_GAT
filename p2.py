import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re  # Import the re module for regular expressions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Load JSON data
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Extract functions and labels from JSON
def extract_functions_and_labels(data):
    functions = {}
    labels = []
    for file_id, content in data['fileid'].items():
        functions[file_id] = content['code']
        labels.append(content['label'])
    return functions, labels

# Generate embeddings using Word2Vec, TextCNN, and Transformer
def generate_feature_embeddings(functions):
    tokenized_functions = [re.findall(r'\w+', func) for func in functions.values()]
    word2vec_model = Word2Vec(sentences=tokenized_functions, vector_size=100, window=5, min_count=1, workers=4)
    embeddings = {name: word2vec_model.wv[re.findall(r'\w+', func)] for name, func in functions.items()}

    textcnn = TextCNN(embedding_dim=100, num_filters=64, kernel_sizes=[3, 4, 5])
    transformer = TransformerEncoder(embedding_dim=64 * 3, nhead=4, num_layers=2)
    feature_embeddings = {}

    for func_name, func_embedding in embeddings.items():
        func_embedding_tensor = torch.tensor(func_embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        textcnn_out = textcnn(func_embedding_tensor)
        transformer_out = transformer(textcnn_out.unsqueeze(1))  # Transformer expects (batch, sequence, embedding) format
        feature_embeddings[func_name] = transformer_out.squeeze(0).detach().numpy()

    return feature_embeddings

# Build function graph
def build_function_graph(functions):
    graph = nx.DiGraph()
    function_names = list(functions.keys())
    for func_name in function_names:
        graph.add_node(func_name)
    for func_name, content in functions.items():
        for target_func in function_names:
            if target_func in content and target_func != func_name:
                graph.add_edge(func_name, target_func)
    return graph

# Prepare data for GCN
def prepare_gcn_data(feature_embeddings, graph):
    node_features = []
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
    for node in graph.nodes:
        if node in feature_embeddings:
            node_features.append(feature_embeddings[node])
        else:
            node_features.append(np.zeros(192))  # Default if embedding missing

    edge_index = []
    for src, dst in graph.edges:
        edge_index.append([node_mapping[src], node_mapping[dst]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index
    )
    return data

# Train and evaluate GCN
def train_and_evaluate_gcn(data_list, labels, num_epochs=10, learning_rate=0.01):
    train_data, test_data, train_labels, test_labels = train_test_split(data_list, labels, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    model = SolidityGNN(in_channels=192, hidden_channels=64, out_channels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, torch.tensor(train_labels, dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    accuracy = accuracy_score(test_labels, all_preds)
    logging.info(f'Test Accuracy: {accuracy}')

# Main function


# Main function
def main():
    json_path = 'data_T.json'
    data = load_json_data(json_path)
    functions, labels = extract_functions_and_labels(data)
    print("Labels:", labels)
    feature_embeddings = generate_feature_embeddings(functions)
    graph = build_function_graph(functions)
    
    # Prepare data for each contract
    data_list = [prepare_gcn_data(feature_embeddings, graph)]
    
    # Debugging: Print lengths
    print(f"Number of data samples: {len(data_list)}")
    print(f"Number of labels: {len(labels)}")
    
    # Ensure lengths match
    if len(data_list) != len(labels):
        raise ValueError("Mismatch between number of data samples and labels.")
    
  #  train_and_evaluate_gcn(data_list, labels)

if __name__ == "__main__":
    main()