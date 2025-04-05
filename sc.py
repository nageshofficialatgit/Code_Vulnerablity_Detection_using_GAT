import os
import re
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from slither.slither import Slither

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
        # Input is already in (batch, sequence, embedding) format
        return self.transformer(x).mean(dim=1)  # Mean pooling over sequence

# Function to extract functions from Solidity files
def extract_functions_from_file(sol_file_path):
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

    return functions

# Generate Word2Vec embeddings
def generate_embeddings(functions):
    tokenized_functions = [re.findall(r'\w+', func) for func in functions.values()]
    model = Word2Vec(sentences=tokenized_functions, vector_size=100, window=5, min_count=1, workers=4)
    embeddings = {name: model.wv[re.findall(r'\w+', func)] for name, func in functions.items()}
    return embeddings

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

# Combine TextCNN and Transformer to generate final feature embeddings
def generate_feature_embeddings(embeddings, embedding_dim=100, num_filters=64, kernel_sizes=[3, 4, 5], nhead=4, num_layers=2):
    textcnn = TextCNN(embedding_dim=embedding_dim, num_filters=num_filters, kernel_sizes=kernel_sizes)
    transformer = TransformerEncoder(embedding_dim=num_filters * len(kernel_sizes), nhead=nhead, num_layers=num_layers)
    feature_embeddings = {}

    for func_name, func_embedding in embeddings.items():
        func_embedding_tensor = torch.tensor(func_embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        textcnn_out = textcnn(func_embedding_tensor)
        transformer_out = transformer(textcnn_out.unsqueeze(1))  # Transformer expects (batch, sequence, embedding) format
        feature_embeddings[func_name] = transformer_out.squeeze(0).detach().numpy()

    return feature_embeddings

# Process Solidity files
def process_sol_files(cache_folder):
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

            print(f"Processed {sol_file}: Graph and feature embeddings saved in {output_folder}")

# Run the script
cache_folder = r"D:\acadmics\sem 5\Innovation paper\IJCAI IMPLE\seems\seems\cache"
process_sol_files(cache_folder)
