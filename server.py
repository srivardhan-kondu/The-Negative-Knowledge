"""
MindGap — Flask REST API Backend
GNN-Based Research Gap Discovery for Mental Health Knowledge Graphs

Endpoints:
    GET  /api/health       — health check, confirms model is loaded
    GET  /api/metrics      — ROC-AUC, graph stats, dataset info, architecture
    GET  /api/predictions  — top-20 global GNN-predicted research gaps
    POST /api/search       — {"query": "anxiety", "top_k": 10} → ranked predictions

Run:
    cd /path/to/Major\ Project
    source venv/bin/activate
    python server.py
"""

import os
import pickle
import sqlite3
import numpy as np

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)  # Allow all origins — change to specific domain in production

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


# ─── GNN Architecture (must match train_gnn.py exactly) ──────────────────────
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, h, out_dim, dropout=0.45):
        super().__init__()
        self.bn_in = torch.nn.BatchNorm1d(in_dim)
        self.proj = torch.nn.Linear(in_dim, h)
        self.conv1 = GCNConv(h, h)
        self.bn1 = torch.nn.BatchNorm1d(h)
        self.conv2 = GCNConv(h, out_dim)
        self.bn2 = torch.nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.bn_in(x)
        x = F.relu(self.proj(x))
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, self.dropout, self.training)
        x = self.bn2(self.conv2(x, edge_index, edge_weight))
        return x


class BilinearDecoder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, z, edge_index):
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        return (z_src @ self.W * z_dst).sum(dim=1) + self.bias


class HybridDecoder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = BilinearDecoder(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(32, 1),
        )

    def forward(self, z, edge_index):
        bilinear_score = self.bilinear(z, edge_index)
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        mlp_score = self.mlp(
            torch.cat([z_src, z_dst, z_src * z_dst], dim=1)
        ).squeeze()
        return bilinear_score + mlp_score


# ─── Global model state (loaded once on startup) ──────────────────────────────
_ctx = {}


def load_model():
    """Load graph, GNN model, compute embeddings, cache everything."""
    if _ctx:
        return  # Already loaded

    print("🔄 Loading GNN model and knowledge graph...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Knowledge graph
    with open(os.path.join(DATA, "mental_health_graph.pkl"), "rb") as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    # PyG graph
    splits = torch.load(
        os.path.join(DATA, "pyg_graph_splits.pt"), weights_only=False
    )
    train_data = splits["train_data"].to(device)
    in_dim = train_data.x.shape[1]

    # GNN model
    model = GCNEncoder(in_dim, h=128, out_dim=64, dropout=0.45).to(device)
    decoder = HybridDecoder(64).to(device)
    checkpoint = torch.load(
        os.path.join(DATA, "gnn_model.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    decoder.eval()

    # Node embeddings
    with torch.no_grad():
        z = model(train_data.x, train_data.edge_index, train_data.edge_weight)

    # ROC-AUC — evaluated on HELD-OUT TEST edges (not training edges)
    test_data = splits["test_data"].to(device)
    with torch.no_grad():
        test_scores = torch.sigmoid(
            decoder(z, test_data.edge_label_index)
        ).cpu().numpy()
    test_labels = test_data.edge_label.cpu().numpy()
    roc_auc = roc_auc_score(test_labels, test_scores)

    # DB stats
    conn = sqlite3.connect(os.path.join(DATA, "mindgap.db"))
    cur = conn.cursor()
    cur.execute("SELECT source, COUNT(*) FROM papers GROUP BY source")
    source_stats = dict(cur.fetchall())
    conn.close()

    # Cache globally
    _ctx.update(
        G=G, nodes=nodes, node_index=node_index,
        z=z, decoder=decoder, device=device,
        roc_auc=roc_auc, source_stats=source_stats,
        train_data=train_data, in_dim=in_dim,
    )
    print(f"✅ Model loaded | Nodes: {len(nodes)} | ROC-AUC: {roc_auc:.4f}")


def score_pair(u, v):
    """Return GNN link-prediction score for node pair (u, v)."""
    ui = _ctx["node_index"][u]
    vi = _ctx["node_index"][v]
    ei = torch.tensor([[ui], [vi]], device=_ctx["device"])
    with torch.no_grad():
        return torch.sigmoid(_ctx["decoder"](_ctx["z"], ei)).item()


# ─── Startup ──────────────────────────────────────────────────────────────────
with app.app_context():
    load_model()


# ─── Frontend Serving ─────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    """Serve the frontend SPA."""
    return send_from_directory(app.static_folder, "index.html")


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_ctx),
        "nodes": len(_ctx.get("nodes", [])),
    })


@app.route("/api/metrics")
def metrics():
    """Return model metrics, graph stats, dataset info, architecture."""
    G = _ctx["G"]
    node_degrees = dict(G.degree())
    avg_degree = sum(node_degrees.values()) / len(_ctx["nodes"])

    return jsonify({
        "roc_auc": round(_ctx["roc_auc"], 6),
        "roc_auc_pct": f"{_ctx['roc_auc']:.2%}",
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": round(float(nx_density(G)), 6),
            "avg_degree": round(avg_degree, 2),
        },
        "dataset": {
            "total_papers": sum(_ctx["source_stats"].values()),
            "sources": _ctx["source_stats"],
        },
        "architecture": {
            "encoder": "GCNEncoder",
            "decoder": "HybridDecoder (Bilinear + MLP)",
            "input_dim": _ctx["in_dim"],
            "hidden_dim": 128,
            "embedding_dim": 64,
            "dropout": 0.45,
            "optimizer": "Adam",
            "lr": 0.003,
            "weight_decay": "5e-4",
            "max_epochs": 600,
            "early_stop_patience": 80,
        },
    })


def nx_density(G):
    """Compute graph density without importing networkx at top level."""
    import networkx as nx
    return nx.density(G)


@app.route("/api/graph_data")
def graph_data():
    """Return complete 3D graph layout (nodes, edges, positions, colors)."""
    G = _ctx["G"]
    nodes = _ctx["nodes"]
    
    import networkx as nx
    
    # Check if we already cached the 3D layout to save computation
    if "pos_3d" not in _ctx:
        # Match original script strictly: spring_layout(dim=3, k=2.0, iterations=200, seed=42)
        _ctx["pos_3d"] = nx.spring_layout(G, dim=3, k=2.0, iterations=200, seed=42)
        
    pos_3d = _ctx["pos_3d"]
    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1

    nodes_data = []
    node_idx = {}
    
    for i, n in enumerate(nodes):
        node_idx[n] = i
        degree = node_degrees[n]
        # INCREASED SIZE: 15-45 (was 8-28) from original script
        size = 15 + (degree / max_degree) * 30
        
        # Professional gradient: dark blue to bright cyan
        intensity = degree / max_degree
        r = int(40 + intensity * 70)   # 40-110
        g = int(150 + intensity * 105) # 150-255
        b = int(200 + intensity * 55)  # 200-255
        opacity = 0.85 + (degree / max_degree) * 0.15
        
        nodes_data.append({
            "id": n,
            "x": pos_3d[n][0],
            "y": pos_3d[n][1],
            "z": pos_3d[n][2],
            "size": size,
            "color": f"rgba({r}, {g}, {b}, {opacity})",
            "degree": degree
        })

    edges_data = []
    for u, v in G.edges():
        edges_data.append([node_idx[u], node_idx[v]])

    return jsonify({
        "nodes": nodes_data,
        "edges": edges_data
    })


@app.route("/api/predictions")
def predictions():
    """Return top-20 globally predicted missing links."""
    G = _ctx["G"]
    nodes = _ctx["nodes"]
    top_k = int(request.args.get("top_k", 20))
    n_samples = int(request.args.get("n_samples", 15000))

    rng = np.random.default_rng(42)
    preds = []
    for _ in range(n_samples):
        a, b = rng.choice(nodes, 2, replace=False)
        if not G.has_edge(a, b):
            preds.append((score_pair(a, b), a, b))

    preds.sort(reverse=True)
    return jsonify([
        {"score": round(p, 4), "score_pct": f"{p:.1%}", "node_a": u, "node_b": v}
        for p, u, v in preds[:top_k]
    ])


@app.route("/api/search", methods=["POST"])
def search():
    """
    Search for GNN-predicted research gaps involving a concept.
    Body: {"query": "anxiety", "top_k": 10}
    """
    body = request.get_json(force=True) or {}
    query = body.get("query", "").strip().lower()
    top_k = int(body.get("top_k", 10))

    if not query:
        return jsonify({"error": "query is required"}), 400

    G = _ctx["G"]
    nodes = _ctx["nodes"]

    # Find matching nodes
    matches = [n for n in nodes if query in n.lower()]

    if not matches:
        return jsonify({
            "query": query,
            "matches_found": 0,
            "results": [],
            "message": f"No nodes found matching '{query}' in the mental health knowledge graph.",
        })

    # Limit to top-3 most connected matching nodes
    matches = sorted(matches, key=lambda n: G.degree(n), reverse=True)[:3]

    all_results = []
    for seed_node in matches:
        neighbors = set(G.neighbors(seed_node))
        neighbors.add(seed_node)

        candidates = []
        for target in nodes:
            if target not in neighbors:
                s = score_pair(seed_node, target)
                candidates.append({"score": round(s, 4), "score_pct": f"{s:.1%}", "node": target})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        all_results.append({
            "seed_node": seed_node,
            "degree": G.degree(seed_node),
            "predictions": candidates[:top_k],
        })

    return jsonify({
        "query": query,
        "matches_found": len(matches),
        "results": all_results,
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"\n🚀 MindGap API starting on http://localhost:{port}")
    print(f"   Frontend: http://localhost:{port}/")
    print(f"   API docs: http://localhost:{port}/api/health\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
