import pickle
import random
import numpy as np
import networkx as nx
from itertools import combinations

# load graph
with open("data/mental_health_graph.pkl", "rb") as f:
    G = pickle.load(f)

# load embeddings
from gensim.models import KeyedVectors
emb = KeyedVectors.load("data/node_embeddings.wv")

def get_vec(node):
    return emb[node] if node in emb else None


# -------- POSITIVE SAMPLES (REAL EDGES) --------
positive_pairs = list(G.edges())
positive_features = []
positive_labels = []

for u, v in positive_pairs:
    u_vec = get_vec(u)
    v_vec = get_vec(v)
    if u_vec is not None and v_vec is not None:
        feature = np.concatenate([u_vec, v_vec])
        positive_features.append(feature)
        positive_labels.append(1)


# -------- NEGATIVE SAMPLES (FAKE EDGES) --------
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
        feature = np.concatenate([u_vec, v_vec])
        negative_features.append(feature)
        negative_labels.append(0)


# -------- COMBINE --------
X = np.vstack([positive_features, negative_features])
y = np.array(positive_labels + negative_labels)

print("Dataset built.")
print("Total samples:", len(y))
print("Positive links:", sum(y))
print("Negative links:", len(y) - sum(y))

# save dataset
import pickle
with open("data/link_data.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("Saved to data/link_data.pkl")
