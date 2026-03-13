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
        z=z, model=model, decoder=decoder, device=device,
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
        
        # Academic gradient: muted blue to vivid indigo
        intensity = degree / max_degree
        r = int(30 + intensity * 90)   # 30-120
        g = int(80 + intensity * 80)   # 80-160
        b = int(180 + intensity * 75)  # 180-255
        opacity = 0.80 + (degree / max_degree) * 0.20
        
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
    """Return top-20 globally predicted missing links (batch-scored for speed)."""
    G = _ctx["G"]
    nodes = _ctx["nodes"]
    node_index = _ctx["node_index"]
    top_k = int(request.args.get("top_k", 20))
    n_samples = int(request.args.get("n_samples", 15000))

    rng = np.random.default_rng(42)

    # 1. Sample candidate pairs that are NOT already connected
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < n_samples and attempts < n_samples * 3:
        attempts += 1
        i, j = rng.integers(0, len(nodes), size=2)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        a, b = nodes[i], nodes[j]
        if not G.has_edge(a, b):
            pairs.append((i, j, a, b))
            seen.add(key)

    if not pairs:
        return jsonify([])

    # 2. Batch score ALL pairs in one forward pass
    src_idx = torch.tensor([p[0] for p in pairs], device=_ctx["device"])
    dst_idx = torch.tensor([p[1] for p in pairs], device=_ctx["device"])
    edge_index = torch.stack([src_idx, dst_idx], dim=0)

    with torch.no_grad():
        scores = torch.sigmoid(_ctx["decoder"](_ctx["z"], edge_index)).cpu().numpy()

    # 3. Sort and return top_k
    ranked = sorted(zip(scores, pairs), key=lambda x: x[0], reverse=True)
    return jsonify([
        {"score": round(float(s), 4), "score_pct": f"{s:.1%}", "node_a": p[2], "node_b": p[3]}
        for s, p in ranked[:top_k]
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


# ─── Explainability Helpers ───────────────────────────────────────────────────

def _gradient_feature_attribution(u_name, v_name):
    """
    GNNExplainer-style gradient attribution.
    Backprop the link score through the GCN encoder to get per-feature importance
    for both nodes, then split into Node2Vec (topology) vs SciBERT (semantics).
    """
    ui = _ctx["node_index"][u_name]
    vi = _ctx["node_index"][v_name]
    device = _ctx["device"]
    model = _ctx.get("model")
    decoder = _ctx["decoder"]
    train_data = _ctx["train_data"]

    if model is None:
        # Fallback: use pre-computed embeddings if model ref not cached
        return None

    # Enable gradient tracking on input features
    x = train_data.x.detach().clone().requires_grad_(True)
    edge_index = train_data.edge_index
    edge_weight = train_data.edge_weight

    # Forward pass (with gradients)
    model.train()  # enable grad flow through dropout/batchnorm
    z = model(x, edge_index, edge_weight)
    ei = torch.tensor([[ui], [vi]], device=device)
    score = torch.sigmoid(decoder(z, ei))
    score.backward()
    model.eval()

    grad_u = x.grad[ui].detach().cpu().numpy()
    grad_v = x.grad[vi].detach().cpu().numpy()

    # Feature importance = |gradient| × |feature value|
    feat_u = train_data.x[ui].detach().cpu().numpy()
    feat_v = train_data.x[vi].detach().cpu().numpy()
    importance_u = np.abs(grad_u) * np.abs(feat_u)
    importance_v = np.abs(grad_v) * np.abs(feat_v)
    combined = importance_u + importance_v

    # Split: first 128 dims = Node2Vec (topology), remaining = SciBERT (semantics)
    n2v_dim = 128
    topo_imp = float(combined[:n2v_dim].sum())
    sem_imp = float(combined[n2v_dim:].sum())
    total = topo_imp + sem_imp + 1e-10

    # Top feature dimensions
    top_dims = np.argsort(combined)[::-1][:20]
    top_features = []
    for d in top_dims:
        top_features.append({
            "dim": int(d),
            "importance": round(float(combined[d]), 6),
            "source": "Node2Vec (topology)" if d < n2v_dim else "SciBERT (semantics)",
        })

    return {
        "topology_pct": round(topo_imp / total, 4),
        "semantics_pct": round(sem_imp / total, 4),
        "topology_raw": round(topo_imp, 6),
        "semantics_raw": round(sem_imp, 6),
        "top_features": top_features[:10],
    }


def _influential_neighbors(target_node, other_node, top_k=5):
    """
    For each neighbor of target_node, check how similar its GNN embedding is
    to other_node's embedding. High similarity = this neighbor is a 'bridge'
    that makes the GNN believe a link should exist.
    """
    G = _ctx["G"]
    z = _ctx["z"]
    node_index = _ctx["node_index"]

    neighbors = list(G.neighbors(target_node))
    if not neighbors:
        return []

    other_idx = node_index[other_node]
    z_other = z[other_idx]

    results = []
    for nb in neighbors:
        nb_idx = node_index[nb]
        z_nb = z[nb_idx]
        cos_sim = float(F.cosine_similarity(z_nb.unsqueeze(0), z_other.unsqueeze(0)).item())
        results.append({
            "node": nb,
            "relevance": round(cos_sim, 4),
            "degree": G.degree(nb),
        })

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]


def _confidence_level(score):
    """Map a prediction score to a human-readable confidence level."""
    if score >= 0.80:
        return "very_strong", "Very Strong — High-priority research direction. The model is highly confident this connection should exist."
    elif score >= 0.70:
        return "strong", "Strong — Likely real research gap. This connection has substantial structural and semantic evidence."
    elif score >= 0.60:
        return "moderate", "Moderate — Worth investigating. The model sees meaningful but not overwhelming evidence."
    else:
        return "weak", "Weak — Speculative. The signal is only slightly above random chance (50%)."


# ─── Explainability Endpoints ─────────────────────────────────────────────────

@app.route("/api/explain", methods=["POST"])
def explain():
    """
    GNNExplainer-style explanation for a predicted link.
    Body: {"node_a": "cbt", "node_b": "insomnia"}
    Returns gradient attribution, common neighbors, shortest path,
    embedding similarity, influential neighbors, paper evidence.
    """
    import networkx as nx

    body = request.get_json(force=True) or {}
    node_a = body.get("node_a", "").strip().lower()
    node_b = body.get("node_b", "").strip().lower()

    if not node_a or not node_b:
        return jsonify({"error": "node_a and node_b are required"}), 400

    G = _ctx["G"]
    nodes = _ctx["nodes"]
    node_index = _ctx["node_index"]
    z = _ctx["z"]

    # Find exact match (nodes are lowercase in graph)
    if node_a not in node_index or node_b not in node_index:
        return jsonify({"error": f"One or both nodes not found in graph."}), 404

    # 1. Prediction score
    score = score_pair(node_a, node_b)

    # 2. Gradient-based feature attribution (topology vs semantics)
    attribution = _gradient_feature_attribution(node_a, node_b)

    # 3. Common neighbors
    neighbors_a = set(G.neighbors(node_a))
    neighbors_b = set(G.neighbors(node_b))
    common = sorted(neighbors_a & neighbors_b, key=lambda n: G.degree(n), reverse=True)
    common_data = [{"node": n, "degree": G.degree(n)} for n in common[:10]]

    # 4. Shortest path
    try:
        path = nx.shortest_path(G, node_a, node_b)
        path_length = len(path) - 1
    except nx.NetworkXNoPath:
        path = []
        path_length = -1

    # 5. Embedding similarity (cosine)
    z_a = z[node_index[node_a]]
    z_b = z[node_index[node_b]]
    emb_sim = float(F.cosine_similarity(z_a.unsqueeze(0), z_b.unsqueeze(0)).item())

    # 6. Influential neighbors (bridge concepts)
    inf_neighbors_a = _influential_neighbors(node_a, node_b, top_k=5)
    inf_neighbors_b = _influential_neighbors(node_b, node_a, top_k=5)

    # 7. Paper evidence from DB
    paper_evidence = {"papers_a": 0, "papers_b": 0, "titles_a": [], "titles_b": []}
    try:
        conn = sqlite3.connect(os.path.join(DATA, "mindgap.db"))
        cur = conn.cursor()
        for label, node_name, key_count, key_titles in [
            ("a", node_a, "papers_a", "titles_a"),
            ("b", node_b, "papers_b", "titles_b"),
        ]:
            cur.execute(
                "SELECT DISTINCT p.title FROM papers p "
                "JOIN entities e ON e.paper_id = p.paper_id "
                "WHERE LOWER(e.entity) = ? LIMIT 50",
                (node_name,)
            )
            titles = [row[0] for row in cur.fetchall()]
            paper_evidence[key_count] = len(titles)
            paper_evidence[key_titles] = titles[:5]
        conn.close()
    except Exception:
        pass

    # 8. Confidence level
    conf_level, conf_desc = _confidence_level(score)

    return jsonify({
        "node_a": node_a,
        "node_b": node_b,
        "score": round(score, 4),
        "score_pct": f"{score:.1%}",
        "confidence_level": conf_level,
        "confidence_description": conf_desc,
        "feature_attribution": attribution,
        "common_neighbors": common_data,
        "common_neighbor_count": len(common),
        "shortest_path": path,
        "shortest_path_length": path_length,
        "embedding_similarity": round(emb_sim, 4),
        "influential_neighbors_a": inf_neighbors_a,
        "influential_neighbors_b": inf_neighbors_b,
        "paper_evidence": paper_evidence,
        "node_a_degree": G.degree(node_a),
        "node_b_degree": G.degree(node_b),
    })


@app.route("/api/node_profile")
def node_profile():
    """Return detailed profile for a single node."""
    node_name = request.args.get("node", "").strip().lower()
    if not node_name or node_name not in _ctx["node_index"]:
        return jsonify({"error": "Node not found"}), 404

    G = _ctx["G"]
    degree = G.degree(node_name)
    neighbors = sorted(G.neighbors(node_name), key=lambda n: G.degree(n), reverse=True)

    # Category from DB
    category = None
    paper_count = 0
    try:
        conn = sqlite3.connect(os.path.join(DATA, "mindgap.db"))
        cur = conn.cursor()
        cur.execute(
            "SELECT category FROM entities WHERE LOWER(entity) = ? AND category IS NOT NULL LIMIT 1",
            (node_name,)
        )
        row = cur.fetchone()
        if row:
            category = row[0]
        cur.execute(
            "SELECT COUNT(DISTINCT paper_id) FROM entities WHERE LOWER(entity) = ?",
            (node_name,)
        )
        paper_count = cur.fetchone()[0]
        conn.close()
    except Exception:
        pass

    return jsonify({
        "node": node_name,
        "degree": degree,
        "category": category,
        "paper_count": paper_count,
        "top_neighbors": [{"node": n, "degree": G.degree(n)} for n in neighbors[:10]],
        "total_neighbors": len(neighbors),
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"\n🚀 MindGap API starting on http://localhost:{port}")
    print(f"   Frontend: http://localhost:{port}/")
    print(f"   API docs: http://localhost:{port}/api/health\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
