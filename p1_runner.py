import logging
import numpy as np
from p1 import ContractFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define the path to the JSON file containing contract data
    json_path = 'data_T.json'

    # Initialize the ContractFeatureExtractor with the JSON path
    feature_extractor = ContractFeatureExtractor(json_path)

    # Process contracts to extract features
    logging.info("Processing contracts to extract features...")
    contracts = feature_extractor.process_contracts()

    # Create communities based on similarity
    logging.info("Creating communities...")
    communities, community_graphs = feature_extractor.create_communities(similarity_threshold=0.8)

    # Analyze the communities
    logging.info("Analyzing communities...")
    feature_extractor.analyze_communities(communities, community_graphs)

    # Generate community embeddings
    logging.info("Generating community embeddings...")
    community_embeddings = feature_extractor.get_community_embeddings(communities, community_graphs)

    # Save the community embeddings to a .npy file
    output_path = "community_embeddings.npy"
    np.save(output_path, community_embeddings)
    logging.info(f"Community embeddings saved to {output_path}")

    # Analyze the generated embeddings
    logging.info("Analyzing community embeddings...")
    feature_extractor.analyze_embeddings(community_embeddings)

    # Assuming labels are available in the JSON file, classify contracts
    data = feature_extractor.load_json_data()
    if 'label' in data['fileid'].values():
        labels = [content['label'] for content in data['fileid'].values()]
        logging.info("Classifying contracts...")
        feature_extractor.classify_contracts(community_embeddings, labels)

if __name__ == "__main__":
    main()