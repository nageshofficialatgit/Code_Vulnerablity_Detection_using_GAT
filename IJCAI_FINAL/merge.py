import pandas as pd
import os

# Prompt user for input file paths
embedding_path = input("Enter the path to embedding.csv: ").strip()
label_path = input("Enter the path to label.csv: ").strip()

try:
    # Load the CSV files
    embedding_df = pd.read_csv(embedding_path)
    label_df = pd.read_csv(label_path)
    
    # Ensure column names are consistent
    embedding_df.rename(columns={'Sol File Name': 'File name'}, inplace=True)
    label_df.rename(columns={'filename': 'File name'}, inplace=True)
    
    # Remove '.sol' from filenames in label.csv
    label_df['File name'] = label_df['File name'].str.replace('.sol', '', regex=False)
    
    # Merge dataframes on 'File name'
    merged_df = pd.merge(embedding_df, label_df[['File name', 'Label']], on='File name', how='inner')
    
    # Construct output path in the same directory as label.csv
    output_path = os.path.join(os.path.dirname(label_path), 'embedding_with_labels.csv')
    
    # Save the result
    merged_df.to_csv(output_path, index=False)
    print(f"File successfully saved to: {output_path}")
    
except Exception as e:
    print(f"An error occurred: {e}")
