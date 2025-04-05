import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

def read_csv_to_dict(file_path, strip_extension=False):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        if strip_extension:
            return {row['Filename'].replace('.sol', ''): row for row in reader}
        else:
            return {row['Filename']: row for row in reader}

def merge_csvs(file1, file2, output_file):
    data1 = read_csv_to_dict(file1)
    data2 = read_csv_to_dict(file2, strip_extension=True)

    # Debug: Print the number of entries in each file
    print(f"Number of entries in {file1}: {len(data1)}")
    print(f"Number of entries in {file2}: {len(data2)}")

    merged_data = []

    for filename in data1:
        if filename in data2:
            merged_row = {
                'Filename': filename,
                'Contract_Embedding': data1[filename]['Contract_Embedding'],
                'Community_Embedding': data2[filename]['Community Embedding'],
                'Label': data1[filename]['Label']
            }
            merged_data.append(merged_row)

    # Debug: Print the number of merged entries
    print(f"Number of merged entries: {len(merged_data)}")

    with open(output_file, mode='w', newline='') as file:
        fieldnames = ['Filename', 'Contract_Embedding', 'Community_Embedding', 'Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

def main():
    # File paths
    contract_file = 'D:/acadmics/sem 5/Innovation paper/IJCAI IMPLE/seems/seems/community_final/embedding_with_labels.csv'
    community_file = 'D:/acadmics/sem 5/Innovation paper/IJCAI IMPLE/seems/seems/community_final/community_embeddings.csv'
    merged_file = 'merged_embeddings.csv'

    # Merge CSVs
    merge_csvs(contract_file, community_file, merged_file)
    print(f'Merged CSV saved to {merged_file}')

    # Load merged CSV
    with open(merged_file, mode='r') as file:
        reader = csv.DictReader(file)
        merged_data = [row for row in reader]

    # Check if merged data is empty
    if not merged_data:
        print("Error: Merged data is empty. Ensure that there are matching filenames in both CSV files.")
        return

    # Convert embeddings from strings to lists
    for row in merged_data:
        row['Contract_Embedding'] = eval(row['Contract_Embedding'])
        row['Community_Embedding'] = eval(row['Community_Embedding'])

    # Prepare data
    contract_embeddings = torch.tensor([row['Contract_Embedding'] for row in merged_data], dtype=torch.float32)
    community_embeddings = torch.tensor([row['Community_Embedding'] for row in merged_data], dtype=torch.float32)

    # Debug: Print shapes
    print(f"Contract Embeddings Shape: {contract_embeddings.shape}")
    print(f"Community Embeddings Shape: {community_embeddings.shape}")

    # Check if tensors are not empty
    if contract_embeddings.size(0) == 0 or community_embeddings.size(0) == 0:
        print("Error: One of the embedding tensors is empty.")
        return

    inputs = torch.cat((contract_embeddings, community_embeddings), dim=1)

    # Debug: Print shape of inputs
    print(f"Inputs Shape: {inputs.shape}")

    # Get user input for epochs
    autoencoder_epochs = int(input("Enter the number of epochs for autoencoder: "))

    # Initialize and train the autoencoder
    input_dim = inputs.size(1)
    hidden_dim = 128  # Adjust as needed
    autoencoder = Autoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in range(autoencoder_epochs):
        optimizer.zero_grad()
        encoded, decoded = autoencoder(inputs)
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Autoencoder Epoch {epoch}, Loss: {loss.item()}')

    # Use the encoded features to create a new CSV
    with torch.no_grad():
        encoded_features, _ = autoencoder(inputs)
    for i, row in enumerate(merged_data):
        row['Encoded_Features'] = encoded_features[i].tolist()

    # Write the encoded features to a new CSV
    with open('encoded_embeddings.csv', mode='w', newline='') as file:
        fieldnames = list(merged_data[0].keys()) + ['Encoded_Features']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

    # Load the new CSV for classification
    with open('encoded_embeddings.csv', mode='r') as file:
        reader = csv.DictReader(file)
        encoded_data = [row for row in reader]

    labels = torch.tensor([float(row['Label']) for row in encoded_data], dtype=torch.float32).unsqueeze(1)
    encoded_features = torch.tensor([eval(row['Encoded_Features']) for row in encoded_data], dtype=torch.float32)

    # Get user input for train-test split
    test_size = float(input("Enter the test size (e.g., 0.2 for 20%): "))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(encoded_features, labels, test_size=test_size, random_state=42)

    # Get user input for MLP epochs
    mlp_epochs = int(input("Enter the number of epochs for MLP classifier: "))

    # Initialize and train the MLP classifier
    mlp = MLPClassifier(hidden_dim, 64, 1)  # Adjust hidden_dim and output_dim as needed
    mlp_criterion = nn.BCEWithLogitsLoss()
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    for epoch in range(mlp_epochs):
        mlp_optimizer.zero_grad()
        outputs = mlp(X_train)
        loss = mlp_criterion(outputs, y_train)
        loss.backward()
        mlp_optimizer.step()
        if epoch % 10 == 0:
            print(f'MLP Epoch {epoch}, Loss: {loss.item()}')

    # Evaluate the model
    with torch.no_grad():
        test_outputs = mlp(X_test)
        predictions = torch.sigmoid(test_outputs).round().numpy()
        true_labels = y_test.numpy()
        100
        print("True labels:", true_labels)
        print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    main()