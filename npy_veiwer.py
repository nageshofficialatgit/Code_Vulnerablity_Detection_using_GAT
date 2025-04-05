import numpy as np

def view_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)  # allow_pickle=True if the file contains objects like dictionaries
        
        # Print the contents
        print("Contents of the .npy file:")
        print(data)
        
        # If the data is a dictionary, print its keys and values
        if isinstance(data, dict):
            print("\nKeys and their corresponding values:")
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print("\nShape of the array:", data.shape)
            print("Data type:", data.dtype)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = "solidity_embeddings.npy"  # Change this to your .npy file path
view_npy_file(file_path)