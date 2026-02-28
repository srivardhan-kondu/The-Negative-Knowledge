import pickle
import torch
import numpy as np
from gensim.models import KeyedVectors
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

# -------- Load Graph --------
with open("data/mental_health_graph.pkl", "rb") as f:
    G = pickle.load(f)

# node list for consistent indexing
nodes = list(G.nodes())
node_index = {n:i for i,n in enumerate(nodes)}

# -------- Load PyG Graph --------
data = torch.load("data/pyg_graph.pt", weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# -------- Define Same GCN --------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(64, 64, 32).to(device)
model.load_state_dict(torch.load("data/gnn_model.pt", map_location=device, weights_only=True))
model.eval()

# -------- Embeddings --------
z = model(data.x, data.edge_index)

def score(u, v):
    ui = node_index[u]
    vi = node_index[v]
    return torch.sigmoid((z[ui] * z[vi]).sum()).item()

# -------- Candidate Links --------
candidates = []
node_list = list(G.nodes())

for i in range(5000):
    u, v = np.random.choice(node_list, 2, replace=False)
    if not G.has_edge(u, v):
        candidates.append((score(u, v), u, v))

# sort descending by probability
candidates.sort(reverse=True)

print("\nTop Predicted Missing Links (GNN):\n")
for p, u, v in candidates[:25]:
    print(f"{u:35}  <-->  {v:35}   score={p:.3f}")
