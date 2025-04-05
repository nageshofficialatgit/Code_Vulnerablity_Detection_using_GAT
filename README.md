# Code_Vulnerablity_Detection_using_GAT
## Overview

**Smart contracts power blockchain applications but carry significant security risks when vulnerable. EFEVD enhances detection by:**

- **Community Feature Extraction:** Grouping contracts with similar semantics and syntax to harvest broader contextual cues.  
- **Contract Graph Construction:** Building function‑level graphs and extracting node and graph features via TextCNN, Transformer, and Graph Neural Networks.  
- **Multi‑Task Learning:** Jointly performing contract‑level vulnerability classification and function‑level localization to learn shared representations.

## Process

### 1. Community Feature Extraction
- **Semantic Embedding:** Embed contract code via Word2Vec + Transformer.  
- **Syntactic Embedding:** Parse Abstract Syntax Trees (ASTs) to capture structural patterns.  
- **Community Detection:** Cluster contracts by combined semantic & syntactic similarity.  
- **Community Graphs:** Connect similar contracts into graphs and extract community features with GNNs.

### 2. Contract Graph Construction
- **Function Embedding:**  
  - Word2Vec for token embeddings  
  - TextCNN for local pattern extraction  
  - Transformer for global context  
- **Graph Assembly:** Nodes = functions; Edges = invocation relationships.  
- **GNN Propagation:** Aggregate node messages to produce an overall contract representation.

### 3. Multi‑Task Downstream Output
- **Graph‑Level Classification:** Predict if a contract is vulnerable.  
- **Node‑Level Localization:** Identify which functions or code segments are vulnerable.  
- **Joint Loss:** Combine graph classification and node classification losses for shared feature
