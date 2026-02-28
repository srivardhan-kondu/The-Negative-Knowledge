"""
GNN Knowledge Graph Dashboard — Streamlit App
Mental Health Research Gap Discovery using Graph Neural Networks

Run with:
    cd /path/to/Major Project
    streamlit run scripts/streamlit_app.py
"""

import os
import sys
import pickle
import sqlite3

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

# ─── Path Setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindGap — GNN Research Gap Discovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30,30,50,0.95), rgba(40,40,65,0.95));
        border: 1px solid rgba(100,180,255,0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #9090b0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #4ade80;
    }

    /* Step cards */
    .step-card {
        background: rgba(40,40,60,0.5);
        border-left: 3px solid #b794f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #d0d0e8;
    }
    .step-label { color: #d4b3ff; font-weight: 600; }

    /* Warning box */
    .warn-box {
        background: rgba(250,200,0,0.1);
        border: 1px solid rgba(250,200,0,0.4);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 12px;
        color: #ffd700;
        margin-top: 14px;
    }

    /* Tab header override */
    .stTabs [data-baseweb="tab"] {
        color: #a0a0c0;
        font-size: 14px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #60d0ff !important; }

    /* Prediction result row */
    .pred-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(30,30,48,0.8);
        border: 1px solid rgba(100,180,255,0.2);
        border-radius: 8px;
        padding: 10px 16px;
        margin: 5px 0;
        font-size: 13px;
    }
    .pred-score {
        color: #4ade80;
        font-weight: 700;
        font-size: 15px;
        min-width: 60px;
        text-align: right;
    }
    .pred-concepts { color: #c0c0e0; }
    .pred-connector { color: #ff6b9d; margin: 0 8px; }

    /* Search input label */
    .search-hint {
        color: #9090b0;
        font-size: 12px;
        margin-bottom: 12px;
    }

    /* Section headers */
    h2, h3 { color: #e0e0f0 !important; }
</style>
""", unsafe_allow_html=True)


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


# ─── Cached Resource Loading ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading knowledge graph & GNN model…")
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Graph
    with open(os.path.join(DATA, "mental_health_graph.pkl"), "rb") as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    # PyG graph splits
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

    # Compute node embeddings
    with torch.no_grad():
        z = model(train_data.x, train_data.edge_index, train_data.edge_weight)

    # ROC-AUC
    neg_ei = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=min(1000, train_data.edge_index.size(1)),
    )
    with torch.no_grad():
        pos_out = torch.sigmoid(decoder(z, train_data.edge_index[:, :1000])).cpu()
        neg_out = torch.sigmoid(decoder(z, neg_ei)).cpu()
    y_true = [1] * len(pos_out) + [0] * len(neg_out)
    y_scores = list(pos_out.numpy()) + list(neg_out.numpy())
    roc_auc = roc_auc_score(y_true, y_scores)

    # DB stats
    conn = sqlite3.connect(os.path.join(DATA, "mindgap.db"))
    cur = conn.cursor()
    cur.execute("SELECT source, COUNT(*) FROM papers GROUP BY source")
    source_stats = dict(cur.fetchall())
    total_papers = sum(source_stats.values())
    conn.close()

    return {
        "G": G,
        "nodes": nodes,
        "node_index": node_index,
        "z": z,
        "decoder": decoder,
        "device": device,
        "roc_auc": roc_auc,
        "source_stats": source_stats,
        "total_papers": total_papers,
        "train_data": train_data,
    }


def score_pair(ctx, u, v):
    """Score a node pair using the GNN decoder."""
    ui = ctx["node_index"][u]
    vi = ctx["node_index"][v]
    ei = torch.tensor([[ui], [vi]], device=ctx["device"])
    with torch.no_grad():
        return torch.sigmoid(ctx["decoder"](ctx["z"], ei)).item()


@st.cache_data(show_spinner="🧮 Generating top-20 global predictions…")
def get_top_global_predictions(_ctx, n_samples=15000, top_k=20):
    """Sample random non-existing pairs and return top-k by GNN score."""
    G = _ctx["G"]
    nodes = _ctx["nodes"]
    preds = []
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        a, b = rng.choice(nodes, 2, replace=False)
        if not G.has_edge(a, b):
            preds.append((score_pair(_ctx, a, b), a, b))
    preds.sort(reverse=True)
    return preds[:top_k]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(ctx):
    with st.sidebar:
        st.markdown("## 🧠 MindGap")
        st.markdown(
            "<span style='color:#9090b0;font-size:13px'>"
            "GNN-based Research Gap Discovery<br>for Mental Health Knowledge Graphs"
            "</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Model performance
        st.markdown("### 🤖 Model Transparency")
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>ROC-AUC Score</div>
                <div class='metric-value'>{ctx['roc_auc']:.2%}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class='metric-card' style='border-color:rgba(180,120,255,0.35)'>
                <div class='metric-label'>Graph Nodes</div>
                <div class='metric-value' style='color:#b794f6'>{len(ctx['nodes'])}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        G = ctx["G"]
        st.markdown(
            f"""<div class='metric-card' style='border-color:rgba(250,100,120,0.35)'>
                <div class='metric-label'>Graph Edges</div>
                <div class='metric-value' style='color:#ff6b9d'>{G.number_of_edges()}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class='metric-card' style='border-color:rgba(100,200,255,0.35)'>
                <div class='metric-label'>Total Papers</div>
                <div class='metric-value' style='color:#60d0ff'>{ctx['total_papers']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # Architecture
        st.markdown("### 📐 Architecture")
        for label, val in [
            ("Model", "GCN Encoder + Hybrid Decoder"),
            ("Layers", "2 GCNConv + BatchNorm"),
            ("Dimensions", "in → 128 → 64"),
            ("Optimizer", "Adam (lr=0.003)"),
            ("Epochs", "up to 600 w/ early stop"),
        ]:
            st.markdown(
                f"<div style='font-size:12px;color:#a0a0c0;margin:3px 0'>"
                f"<b style='color:#8ab4f8'>{label}:</b> {val}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Data sources
        st.markdown("### 📚 Data Sources")
        for source, count in sorted(
            ctx["source_stats"].items(), key=lambda x: x[1], reverse=True
        ):
            st.markdown(
                f"<div style='font-size:12px;color:#a0a0c0;margin:3px 0'>"
                f"<b style='color:#8ab4f8'>{source}:</b> {count} papers</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Methodology
        st.markdown("### 🔬 Methodology")
        for num, title, desc in [
            ("1", "Collection", "Fetch papers from PubMed"),
            ("2", "Extraction", "NLP identifies concepts (scispaCy)"),
            ("3", "Graph", "Link co-occurring concepts"),
            ("4", "Learning", "GNN predicts missing links"),
            ("5", "Discovery", "Rank potential research gaps"),
        ]:
            st.markdown(
                f"<div class='step-card'><span class='step-label'>{num}. {title}:</span> {desc}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='warn-box'>⚠️ <b>Disclaimer:</b> AI-generated hypotheses. "
            "Requires domain expert validation.</div>",
            unsafe_allow_html=True,
        )


# ─── Tab 1: Search Predictions ───────────────────────────────────────────────
def render_search_tab(ctx):
    st.markdown("## 🔍 Search Research Gaps")
    st.markdown(
        "<p class='search-hint'>Type a mental health concept to find under-researched "
        "connections predicted by the GNN model.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search concept",
            placeholder="e.g. anxiety, depression, schizophrenia, trauma…",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.selectbox("Top", [5, 10, 20], index=1, label_visibility="collapsed")

    if not query:
        st.info("💡 Enter a concept above to discover predicted research gaps.")
        return

    G = ctx["G"]
    nodes = ctx["nodes"]
    q = query.strip().lower()

    # Find matching nodes
    matches = [n for n in nodes if q in n.lower()]

    if not matches:
        st.error(
            f"❌ **No nodes found** matching `{query}` in the mental health knowledge graph.\n\n"
            "This concept may not exist in the current training domain. "
            "Multi-domain support (e.g. environmental science) is planned for the next sprint."
        )
        return

    # Clamp results if many matches
    if len(matches) > 3:
        st.warning(
            f"Found **{len(matches)} matching nodes**. Showing predictions for the top 3 most connected."
        )
        matches = sorted(matches, key=lambda n: G.degree(n), reverse=True)[:3]

    for seed_node in matches:
        st.markdown(f"### 🔵 Predictions for: `{seed_node}`")
        degree = G.degree(seed_node)
        st.markdown(
            f"<span style='color:#9090b0;font-size:12px'>Current connections in graph: {degree}</span>",
            unsafe_allow_html=True,
        )

        # Score against all non-neighbors
        candidates = []
        neighbors = set(G.neighbors(seed_node))
        neighbors.add(seed_node)

        with st.spinner(f"Running GNN predictions for '{seed_node}'…"):
            for target in nodes:
                if target not in neighbors:
                    s = score_pair(ctx, seed_node, target)
                    candidates.append((s, target))

        candidates.sort(reverse=True)

        # Render results
        for rank, (s, target) in enumerate(candidates[:top_k], 1):
            bar_width = int(s * 100)
            st.markdown(
                f"""<div class='pred-row'>
                    <span style='color:#9090b0;min-width:28px'>{rank}.</span>
                    <span class='pred-concepts'>
                        <b style='color:#60d0ff'>{seed_node}</b>
                        <span class='pred-connector'>↔</span>
                        <b style='color:#e0e0f0'>{target}</b>
                    </span>
                    <span class='pred-score'>{s:.1%}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)


# ─── Tab 2: 3D Knowledge Graph ────────────────────────────────────────────────
def render_graph_tab(ctx):
    st.markdown("## 🌐 3D Knowledge Graph")
    st.markdown(
        "<p style='color:#9090b0;font-size:13px'>"
        "Rotate · Zoom · Hover over nodes. "
        "Red edges = top-20 AI-predicted research gaps. "
        "Blue/cyan nodes = mental health concepts (larger = more connections)."
        "</p>",
        unsafe_allow_html=True,
    )

    G = ctx["G"]
    nodes = ctx["nodes"]

    top_preds = get_top_global_predictions(ctx)

    with st.spinner("Building 3D layout…"):
        pos_3d = nx.spring_layout(G, dim=3, k=2.0, iterations=200, seed=42)

    node_x = [pos_3d[n][0] for n in nodes]
    node_y = [pos_3d[n][1] for n in nodes]
    node_z = [pos_3d[n][2] for n in nodes]

    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values())

    node_colors, node_sizes = [], []
    for n in nodes:
        deg = node_degrees[n]
        node_sizes.append(12 + (deg / max_degree) * 28)
        intensity = deg / max_degree
        r = int(40 + intensity * 70)
        g = int(150 + intensity * 105)
        b = int(200 + intensity * 55)
        node_colors.append(f"rgba({r},{g},{b},0.9)")

    # Existing edges
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        ex += [pos_3d[u][0], pos_3d[v][0], None]
        ey += [pos_3d[u][1], pos_3d[v][1], None]
        ez += [pos_3d[u][2], pos_3d[v][2], None]

    # Predicted edges
    px, py, pz = [], [], []
    for _, u, v in top_preds:
        px += [pos_3d[u][0], pos_3d[v][0], None]
        py += [pos_3d[u][1], pos_3d[v][1], None]
        pz += [pos_3d[u][2], pos_3d[v][2], None]

    hover_texts = [
        f"<b>{n}</b><br>Connections: {node_degrees[n]}" for n in nodes
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color="rgba(120,120,140,0.12)", width=1),
                hoverinfo="none",
                name="Known Connections",
            ),
            go.Scatter3d(
                x=px, y=py, z=pz,
                mode="lines",
                line=dict(color="rgba(250,80,100,0.9)", width=3.5),
                hoverinfo="none",
                name="AI Predicted Gaps",
            ),
            go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode="markers",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(color="rgba(255,255,255,0.25)", width=1.5),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Mental Health Concepts",
            ),
        ]
    )

    fig.update_layout(
        paper_bgcolor="#0f1117",
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, gridcolor="rgba(80,80,100,0.15)"),
            yaxis=dict(showbackground=False, showticklabels=False, gridcolor="rgba(80,80,100,0.15)"),
            zaxis=dict(showbackground=False, showticklabels=False, gridcolor="rgba(80,80,100,0.15)"),
            bgcolor="#0f1117",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        legend=dict(
            bgcolor="rgba(20,20,35,0.9)",
            bordercolor="rgba(100,180,255,0.3)",
            borderwidth=1,
            font=dict(color="#e0e0e0", size=12),
        ),
        hovermode="closest",
        font=dict(color="#e0e0e0"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # Top predictions table below graph
    st.markdown("### 🔴 Top 20 Predicted Research Gaps")
    st.markdown(
        "<p style='color:#9090b0;font-size:12px'>Sampled 15,000 candidate pairs · sorted by GNN confidence</p>",
        unsafe_allow_html=True,
    )
    for i, (p, u, v) in enumerate(top_preds, 1):
        st.markdown(
            f"""<div class='pred-row'>
                <span style='color:#9090b0;min-width:28px'>{i}.</span>
                <span class='pred-concepts'>
                    <b style='color:#60d0ff'>{u}</b>
                    <span class='pred-connector'>↔</span>
                    <b style='color:#e0e0f0'>{v}</b>
                </span>
                <span class='pred-score'>{p:.1%}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ─── Tab 3: Model Metrics ────────────────────────────────────────────────────
def render_metrics_tab(ctx):
    st.markdown("## 📊 Model Metrics & Dataset Info")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Performance")
        auc = ctx["roc_auc"]
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>ROC-AUC Score</div>
                <div class='metric-value'>{auc:.4f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        # AUC gauge bar
        st.markdown(
            f"""<div style='margin:12px 0;'>
                <div style='font-size:11px;color:#9090b0;margin-bottom:4px'>
                    Model Confidence (0.5 = random, 1.0 = perfect)
                </div>
                <div style='background:rgba(40,40,60,0.7);border-radius:8px;height:16px;width:100%;'>
                    <div style='
                        background:linear-gradient(90deg,#4ade80,#22d3ee);
                        border-radius:8px;
                        height:16px;
                        width:{int((auc-0.5)*200)}%;
                        max-width:100%;
                    '></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### 🗂️ Graph Statistics")
        G = ctx["G"]
        for label, val in [
            ("Nodes (Concepts)", len(ctx["nodes"])),
            ("Edges (Known Links)", G.number_of_edges()),
            ("Total Research Papers", ctx["total_papers"]),
            ("Graph Density", f"{nx.density(G):.5f}"),
            ("Avg Degree", f"{sum(dict(G.degree()).values()) / len(ctx['nodes']):.2f}"),
        ]:
            st.markdown(
                f"""<div style='display:flex;justify-content:space-between;
                    background:rgba(30,30,50,0.6);border-radius:6px;
                    padding:9px 14px;margin:5px 0;font-size:13px;'>
                    <span style='color:#9090b0'>{label}</span>
                    <span style='color:#60d0ff;font-weight:600'>{val}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 📚 Papers by Source")
        # Donut chart
        sources = list(ctx["source_stats"].keys())
        counts = [ctx["source_stats"][s] for s in sources]
        palette = ["#60d0ff", "#4ade80", "#ff6b9d", "#b794f6", "#ffd700"]
        fig_pie = go.Figure(
            go.Pie(
                labels=sources,
                values=counts,
                hole=0.55,
                marker=dict(
                    colors=palette[: len(sources)],
                    line=dict(color="#0f1117", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(color="#e0e0e0", size=12),
            )
        )
        fig_pie.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            showlegend=False,
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 🏗️ Architecture Summary")
        arch_rows = [
            ("Encoder", "GCNEncoder"),
            ("Decoder", "HybridDecoder (Bilinear + MLP)"),
            ("Input Dim", str(ctx["train_data"].x.shape[1])),
            ("Hidden Dim", "128"),
            ("Embedding Dim", "64"),
            ("Dropout", "0.45"),
            ("Optimizer", "Adam · lr=0.003 · wd=5e-4"),
            ("Scheduler", "CosineAnnealingWarmRestarts"),
            ("Max Epochs", "600"),
            ("Early Stop Patience", "80"),
        ]
        for label, val in arch_rows:
            st.markdown(
                f"""<div style='display:flex;justify-content:space-between;
                    background:rgba(30,30,50,0.6);border-radius:6px;
                    padding:8px 14px;margin:4px 0;font-size:12px;'>
                    <span style='color:#9090b0'>{label}</span>
                    <span style='color:#b794f6;font-weight:500'>{val}</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    # Load everything
    ctx = load_all()

    # Sidebar
    render_sidebar(ctx)

    # Header
    st.markdown(
        "<h1 style='color:#e0e0f0;margin-bottom:4px'>🧠 Mental Health Knowledge Graph</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#9090b0;font-size:14px;margin-top:0'>"
        "AI-Powered Research Gap Discovery · GNN-Based Negative Knowledge Detection</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍  Search Predictions", "🌐  3D Knowledge Graph", "📊  Model Metrics"]
    )

    with tab1:
        render_search_tab(ctx)

    with tab2:
        render_graph_tab(ctx)

    with tab3:
        render_metrics_tab(ctx)


if __name__ == "__main__":
    main()
