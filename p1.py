import os
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
import networkx as nx
from collections import defaultdict
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
import csv

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=None, dropout=0.2):
        super(GCNEncoder, self).__init__()
        self.embedding_dim = embedding_dim if embedding_dim is not None else input_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, self.embedding_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class ContractFeatureExtractor:
    def __init__(self, cache_folder):
        self.cache_folder = cache_folder

    def load_contracts_from_folder(self):
        contracts = {}
        for filename in os.listdir(self.cache_folder):
            if filename.endswith('.sol'):
                file_path = os.path.join(self.cache_folder, filename)
                with open(file_path, 'r') as file:
                    contracts[filename] = file.read()
        return contracts

    def get_semantic_features(self, contract_code):
        tokens = contract_code.split()
        model = Word2Vec([tokens], vector_size=100, window=5, min_count=1)
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0).astype(np.float32) if vectors else np.zeros(100, dtype=np.float32)

    def get_syntactic_features(self, contract_code):
        # Simplified syntactic feature extraction
        features = {
            'pragma_statements': contract_code.count('pragma'),
            'contract_definitions': contract_code.count('contract'),
            'function_definitions': contract_code.count('function'),
            'state_variables': contract_code.count('var'),
            'modifiers': contract_code.count('modifier'),
            'events': contract_code.count('event'),
            'structs': contract_code.count('struct'),
            'enums': contract_code.count('enum')
        }
        return np.array(list(features.values()), dtype=np.float32)

    def calculate_similarity(self, contract1_features, contract2_features):
        semantic_sim = np.dot(contract1_features[0], contract2_features[0])
        syntactic_sim = np.dot(contract1_features[1], contract2_features[1])
        return (semantic_sim + syntactic_sim) / 2

    def process_contracts(self):
        contracts = self.load_contracts_from_folder()
        contract_features = {}
        for filename, code in contracts.items():
            semantic_features = self.get_semantic_features(code)
            syntactic_features = self.get_syntactic_features(code)
            contract_features[filename] = (semantic_features, syntactic_features)
        return contract_features

    def create_communities(self, similarity_threshold=0.8):
        contract_features = self.process_contracts()
        communities = defaultdict(list)
        community_id = 0
        assigned_contracts = set()

        for contract1 in contract_features:
            if contract1 in assigned_contracts:
                continue

            current_community = [contract1]
            assigned_contracts.add(contract1)

            for contract2 in contract_features:
                if contract2 not in assigned_contracts:
                    similarity = self.calculate_similarity(
                        contract_features[contract1],
                        contract_features[contract2]
                    )
                    if similarity >= similarity_threshold:
                        current_community.append(contract2)
                        assigned_contracts.add(contract2)

            if current_community:
                communities[community_id] = current_community
                community_id += 1

        community_graphs = {}
        for comm_id, contracts in communities.items():
            G = nx.Graph()
            for contract in contracts:
                G.add_node(contract, features=contract_features[contract])

            for i, contract1 in enumerate(contracts):
                for contract2 in contracts[i+1:]:
                    similarity = self.calculate_similarity(
                        contract_features[contract1],
                        contract_features[contract2]
                    )
                    if similarity >= similarity_threshold:
                        G.add_edge(contract1, contract2, weight=similarity)

            community_graphs[comm_id] = G

        return communities, community_graphs

    def analyze_communities(self, communities, graphs):
        print(f"Total number of communities: {len(communities)}")
        for comm_id, contracts in communities.items():
            print(f"\nCommunity {comm_id}:")
            print(f"Number of contracts: {len(contracts)}")
            print(f"Contracts: {', '.join(contracts)}")
            G = graphs[comm_id]
            print(f"Number of edges: {G.number_of_edges()}")
            print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

    def prepare_graph_data(self, G):
        node_features = []
        node_mapping = {}
        for idx, (node, data) in enumerate(G.nodes(data=True)):
            node_mapping[node] = idx
            semantic = torch.tensor(data['features'][0], dtype=torch.float32)
            syntactic = torch.tensor(data['features'][1], dtype=torch.float32)
            features = torch.cat([semantic, syntactic])
            node_features.append(features)

        node_features = torch.stack(node_features)
        edge_index = []
        edge_weights = []
        for u, v, data in G.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])
            edge_weights.extend([data['weight'], data['weight']])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights
        )

    def get_community_embeddings(self, communities, graphs, device='cuda' if torch.cuda.is_available() else 'cpu',
                                 batch_size=32, epochs=100, learning_rate=0.001):
        community_embeddings = {}

        for comm_id, G in graphs.items():
            if len(G.nodes()) == 0:
                continue

            data = self.prepare_graph_data(G)
            data = data.to(device)
            data.x = data.x.float()
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr.float()

            input_dim = data.x.size(1)
            model = GCNEncoder(
                input_dim=input_dim,
                hidden_dim=2*input_dim,
                embedding_dim=input_dim,
                dropout=0.2
            ).to(device)

            model = model.float()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr)
                loss = F.mse_loss(out, data.x)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f'Community {comm_id}, Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                x = model.conv1(data.x, data.edge_index, data.edge_attr)
                x = F.relu(x)
                community_embedding = torch.mean(x, dim=0)
                community_embeddings[comm_id] = community_embedding.cpu().numpy()

        return community_embeddings

    def analyze_embeddings(self, community_embeddings):
        print("\nCommunity Embedding Analysis:")
        for comm_id, embedding in community_embeddings.items():
            print(f"\nCommunity {comm_id}:")
            print(f"Embedding dimension: {embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"Mean value: {np.mean(embedding):.4f}")
            print(f"Std deviation: {np.std(embedding):.4f}")

    def save_embeddings_to_csv(self, communities, community_embeddings, filename='community_embeddings.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Community Embedding'])
            for comm_id, contracts in communities.items():
                embedding = community_embeddings[comm_id]
                for contract in contracts:
                    writer.writerow([contract, embedding.tolist()])

# Example usage
if __name__ == "__main__":
    cache_folder = r'D:\acadmics\sem 5\Innovation paper\IJCAI IMPLE\seems\seems\cache'
    feature_extractor = ContractFeatureExtractor(cache_folder)
    communities, community_graphs = feature_extractor.create_communities()
    feature_extractor.analyze_communities(communities, community_graphs)
    community_embeddings = feature_extractor.get_community_embeddings(communities, community_graphs)
    feature_extractor.analyze_embeddings(community_embeddings)
    feature_extractor.save_embeddings_to_csv(communities, community_embeddings)