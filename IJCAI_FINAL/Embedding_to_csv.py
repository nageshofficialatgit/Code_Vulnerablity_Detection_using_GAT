import os
import numpy as np
import pandas as pd

def process_solidity_embeddings(npy_path):
    try:
        # Load the .npy file
        data = np.load(npy_path, allow_pickle=True).item()  # Extract dictionary from 0-d array

        # Extract keys and values
        sol_file_names = list(data.keys())
        embeddings = [data[key].tolist() for key in sol_file_names]

        # Create a DataFrame
        df = pd.DataFrame({
            'Sol File Name': sol_file_names,
            'Embedding': [str(embedding) for embedding in embeddings]  # Convert embeddings to strings
        })

        # Determine the output path
        base_name = os.path.splitext(npy_path)[0]  # Remove extension
        output_csv_path = f"{base_name}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(output_csv_path, index=False)
        print(f"CSV file created successfully at: {output_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    npy_path = input("Enter the path to the Solidity embeddings .npy file: ")
    process_solidity_embeddings(npy_path)
