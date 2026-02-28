import pickle
import random
import numpy as np
import networkx as nx
from gensim.models import KeyedVectors

# Load graph
with open("data/mental_health_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load embeddings
emb = KeyedVectors.load("data/node_embeddings.wv")

def get_vec(node):
    return emb[node] if node in emb else None

def compute_graph_features(G, u, v):
    """Compute advanced graph-based features for a node pair"""
    features = []
    
    # Common neighbors
    common = len(list(nx.common_neighbors(G, u, v)))
    features.append(common)
    
    # Jaccard coefficient
    preds = nx.jaccard_coefficient(G, [(u, v)])
    jaccard = next(preds)[2] if preds else 0
    features.append(jaccard)
    
    # Adamic-Adar index
    preds = nx.adamic_adar_index(G, [(u, v)])
    aa = next(preds)[2] if preds else 0
    features.append(aa)
    
    # Preferential attachment
    preds = nx.preferential_attachment(G, [(u, v)])
    pa = next(preds)[2] if preds else 0
    features.append(pa)
    
    # Node degrees
    features.append(G.degree(u))
    features.append(G.degree(v))
    
    return features


# -------- POSITIVE SAMPLES --------
positive_pairs = list(G.edges())
positive_features = []
positive_labels = []

for u, v in positive_pairs:
    u_vec = get_vec(u)
    v_vec = get_vec(v)
    if u_vec is not None and v_vec is not None:
        # Node embeddings (concatenated)
        emb_features = np.concatenate([u_vec, v_vec])
        
        # Embedding-based features (element-wise operations)
        hadamard = u_vec * v_vec  # Element-wise product
        avg = (u_vec + v_vec) / 2  # Average
        l1_dist = np.abs(u_vec - v_vec)  # L1 distance
        l2_dist = np.sqrt((u_vec - v_vec) ** 2)  # L2 distance
        
        # Graph-based features
        graph_features = compute_graph_features(G, u, v)
        
        # Combine all features
        feature = np.concatenate([
            emb_features,
            hadamard,
            avg,
            l1_dist,
            l2_dist,
            graph_features
        ])
        
        positive_features.append(feature)
        positive_labels.append(1)


# -------- NEGATIVE SAMPLES --------
nodes = list(G.nodes())
negative_pairs = set()

while len(negative_pairs) < len(positive_pairs):
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v):
        negative_pairs.add((u, v))

negative_features = []
negative_labels = []

for u, v in negative_pairs:
    u_vec = get_vec(u)
    v_vec = get_vec(v)
    if u_vec is not None and v_vec is not None:
        # Same feature engineering as positive samples
        emb_features = np.concatenate([u_vec, v_vec])
        hadamard = u_vec * v_vec
        avg = (u_vec + v_vec) / 2
        l1_dist = np.abs(u_vec - v_vec)
        l2_dist = np.sqrt((u_vec - v_vec) ** 2)
        graph_features = compute_graph_features(G, u, v)
        
        feature = np.concatenate([
            emb_features,
            hadamard,
            avg,
            l1_dist,
            l2_dist,
            graph_features
        ])
        
        negative_features.append(feature)
        negative_labels.append(0)


# -------- COMBINE --------
X = np.vstack([positive_features, negative_features])
y = np.array(positive_labels + negative_labels)

print("Enhanced dataset built.")
print("Total samples:", len(y))
print("Total features per sample:", X.shape[1])
print("Positive links:", sum(y))
print("Negative links:", len(y) - sum(y))

# Save dataset
with open("data/link_data_enhanced.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("Saved to data/link_data_enhanced.pkl")
