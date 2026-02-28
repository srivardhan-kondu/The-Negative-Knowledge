import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from gensim.models import KeyedVectors

# Load the graph you already built
with open("data/mental_health_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Convert to PyTorch Geometric format manually to avoid metadata slice issues
data = Data()

# ========================================================================
# HYBRID NODE FEATURES: Node2Vec (128-D) + SciBERT (768-D) = 896-D
# Node2Vec captures structural topology (graph shape/connectivity)
# SciBERT captures semantic meaning (actual medical content)
# Combining both gives the model a 360-degree view of each entity.
# ========================================================================
print("Loading SciBERT semantic embeddings...")
scibert_emb = pickle.load(open("data/semantic_embeddings.pkl", "rb"))

print("Loading Node2Vec structural embeddings...")
node2vec_emb = KeyedVectors.load("data/node_embeddings.wv")

# Create a mapping from node names to integer indices
node_list = list(G.nodes())
node_to_idx = {node: idx for idx, node in enumerate(node_list)}

SCIBERT_DIM = 768
NODE2VEC_DIM = 128  # We'll use 128-D Node2Vec

# Build combined node feature matrix
print(f"Building hybrid feature matrix for {len(node_list)} nodes...")
X = []
scibert_covered = 0
node2vec_covered = 0
for node in node_list:
    # SciBERT embedding (768-D)
    if node in scibert_emb:
        sc = scibert_emb[node]
        scibert_covered += 1
    else:
        sc = np.zeros(SCIBERT_DIM)

    # Node2Vec embedding (128-D or pad to 128)
    if node in node2vec_emb:
        nv_raw = node2vec_emb[node]
        # Pad or trim to NODE2VEC_DIM (in case model was trained at 64-D)
        if len(nv_raw) >= NODE2VEC_DIM:
            nv = nv_raw[:NODE2VEC_DIM]
        else:
            nv = np.pad(nv_raw, (0, NODE2VEC_DIM - len(nv_raw)))
        node2vec_covered += 1
    else:
        nv = np.zeros(NODE2VEC_DIM)

    # Concatenate SciBERT + Node2Vec
    X.append(np.concatenate([sc, nv]))

print(f"  SciBERT coverage: {scibert_covered}/{len(node_list)}")
print(f"  Node2Vec coverage: {node2vec_covered}/{len(node_list)}")
print(f"  Combined feature dimension: {len(X[0])}")

# Convert to tensor efficiently
data.x = torch.tensor(np.array(X), dtype=torch.float)

# Convert edges and weights to integer indices and tensors
edge_list = []
edge_weights = []

for u, v, data_dict in G.edges(data=True):
    weight = data_dict.get('weight', 1.0)
    
    # Original direction
    edge_list.append([node_to_idx[u], node_to_idx[v]])
    edge_weights.append(weight)
    
    # Reverse direction for undirected graph
    edge_list.append([node_to_idx[v], node_to_idx[u]])
    edge_weights.append(weight)

data.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Add edge weights to data object
# Normalize weights slightly using log scaling to prevent enormous values dominating
data.edge_weight = torch.tensor(np.log1p(edge_weights), dtype=torch.float)

# Store node name mapping for later use
data.node_names = node_list

import torch_geometric.transforms as T
# Create splits: 80% train, 10% val, 10% test
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False  # We will sample negatives during training
)
train_data, val_data, test_data = transform(data)

torch.save({
    "train_data": train_data,
    "val_data": val_data,
    "test_data": test_data,
    "node_names": node_list
}, "data/pyg_graph_splits.pt")

print("\n✅ PyTorch Geometric graph saved successfully!")
print("\nGraph Statistics:")
print(f"  Number of nodes: {data.x.shape[0]}")
print(f"  Number of edges: {data.edge_index.shape[1]}")
print(f"  Node feature dimension: {data.x.shape[1]}")
print(f"\n{data}")
