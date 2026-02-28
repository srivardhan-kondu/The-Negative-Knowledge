import pickle
import random
import numpy as np
from gensim.models import KeyedVectors
import networkx as nx

# Load graph
with open("data/mental_health_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load embeddings (correct format)
emb = KeyedVectors.load("data/node_embeddings.wv")

# Load best enhanced model + scaler
with open("data/link_model_best.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']

def get_vec(node):
    return emb[node] if node in emb else None

def compute_graph_features(G, u, v):
    """Compute advanced graph-based features"""
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

def build_features(u, v):
    """Build complete feature vector for node pair"""
    u_vec = get_vec(u)
    v_vec = get_vec(v)
    
    if u_vec is None or v_vec is None:
        return None
    
    # Embedding features
    emb_features = np.concatenate([u_vec, v_vec])
    
    # Embedding operations
    hadamard = u_vec * v_vec
    avg = (u_vec + v_vec) / 2
    l1_dist = np.abs(u_vec - v_vec)
    l2_dist = np.sqrt((u_vec - v_vec) ** 2)
    
    # Graph features
    graph_features = compute_graph_features(G, u, v)
    
    # Combine all
    feature = np.concatenate([
        emb_features,
        hadamard,
        avg,
        l1_dist,
        l2_dist,
        graph_features
    ])
    
    return feature

# Sample candidate pairs
print("Searching for potential missing links...")
nodes = list(G.nodes())
candidates = []

for _ in range(5000):
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v):
        features = build_features(u, v)
        if features is not None:
            # Scale features
            features_scaled = scaler.transform([features])
            prob = model.predict_proba(features_scaled)[0][1]
            candidates.append((prob, u, v))

# Sort by probability
candidates.sort(reverse=True)

print("\n" + "="*80)
print("TOP 20 PREDICTED MISSING LINKS (High Confidence)")
print("="*80)
print(f"{'Rank':<6} {'Confidence':<12} {'Entity 1':<30} {'Entity 2':<30}")
print("-"*80)

for i, (p, u, v) in enumerate(candidates[:20], 1):
    print(f"{i:<6} {p:.4f}       {u[:28]:<30} {v[:28]:<30}")

print("="*80)
print(f"\nTotal candidates evaluated: {len(candidates)}")
print(f"Using model: {saved['model_name']} (98.87% accuracy)")
