# 🧠 The Negative Knowledge

### AI-Powered Research Gap Discovery using Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen.svg)](data/graph_credible_ai.html)
[![ROC-AUC](https://img.shields.io/badge/Test_ROC--AUC-74.1%25-success.svg)]()

<p align="center">
  <img src="assets/demo.webp" alt="3D Interactive Visualization" width="800"/>
</p>

---

## 🎯 **What is Negative Knowledge?**

**Negative Knowledge** represents the *unknown unknowns* in scientific research — **connections that should exist but haven't been studied yet**. This project uses cutting-edge AI to discover these hidden research opportunities in medical science.

### The Problem
- 📚 Millions of research papers published annually
- 🔍 Researchers can't read everything
- ❓ Promising research directions remain unexplored
- 💡 Valuable connections between concepts are missed

### Our Solution
Using **Graph Neural Networks (GNNs)** combined with **deep biomedical language models (SciBERT)**, we construct vast knowledge graphs from raw research papers and algorithmically predict which research connections are missing with a robust **74.1% true unseen test accuracy**.

---

## ✨ **Major System Upgrades & Features**

This project has been massively scaled and re-engineered to prevent AI "hallucinations" and provide scientifically rigorous, proven predictions.

### 🤖 **Massive Multi-Source Data Collection**
- **Semantic Scholar & PubMed APIs**: Built a high-throughput fetcher (`fetch_papers_large.py`).
- **Scale**: Expanded from 500+ papers to over **872 mental health research papers** by rapidly querying 20 distinct psychiatric search terms.

### 🧠 **Advanced Hybrid AI Pipeline — 896-Dimensional Feature Vector**

This is the core innovation of the project. Every single research concept (node) in the knowledge graph is represented using a **two-part, 896-dimensional hybrid vector** that encodes both **how the node exists structurally in the graph** AND **what it actually means medically**.

#### Part 1: Node2Vec — Topology-Aware Embeddings (128 Dimensions)

**Node2Vec** (Grover & Leskovec, 2016) performs biased random walks on the knowledge graph to learn a node's "neighborhood identity."

```
For each node n in the graph:
  1. Generate K=200 random walks of length L=20 steps
  2. Bias walk direction using parameters:
       p = 1.0  (return parameter — controls revisit probability)
       q = 0.5  (in-out parameter — controls DFS vs BFS exploration)
  3. Feed all walks into Word2Vec Skip-gram model
  4. Output: 128-dimensional dense vector z_topo ∈ ℝ^128
```

**Result**: Nodes with structurally similar neighborhoods in the graph (even if they have very different names) will have similar `z_topo` vectors.

#### Part 2: SciBERT — Semantic Language Embeddings (768 Dimensions)

**SciBERT** (`allenai/scibert_scivocab_uncased`) is a BERT transformer pre-trained on 1.14 **million scientific papers**. For every node (e.g., `"cognitive behavioral therapy"`), we:

```
Input text → SciBERT Tokenizer
         → 12-Layer Transformer (110M parameters)
         → Extract [CLS] token from final hidden state
         → Output: 768-dimensional dense vector z_sem ∈ ℝ^768
```

**Result**: Even an isolated node with zero graph connections (a new concept with no co-occurrences yet) gets a rich, medically informed vector because SciBERT was trained on 1 million papers.

#### Final Combination — The 896D Vector

```
z_node = CONCATENATE(z_topo, z_sem)
       = [128D Node2Vec | 768D SciBERT]
       = 896-Dimensional Final Feature Vector
```

This means the AI understands:
- `Node2Vec` component → *"CBT is highly connected to Depression and Anxiety in the literature"*
- `SciBERT` component  → *"CBT is a structured psychotherapy that modifies dysfunctional thought patterns"*

---

### 🧬 **State-of-the-Art Neural Architecture**

#### The Encoder — GCN with Anti-Overfitting Safeguards

A 2-Layer Graph Convolutional Network reduces the 896D vectors into compact 64D embeddings by aggregating each node's representation with its neighbors.

```
LAYER 1:
  H¹ = ReLU( BatchNorm( D̃^(-½) Ã D̃^(-½) · X · W₁ ) )
  Then: Dropout(H¹, p=0.45)       ← Randomly zeroes 45% of units per training step

Where:
  X   = Input feature matrix ∈ ℝ^(659 × 896)
  Ã   = Adjacency matrix + Self-loops (A + I)
  D̃   = Degree matrix of Ã
  W₁  = Learnable weight matrix ∈ ℝ^(896 × 128)
  BatchNorm = Normalizes activations for stable training

LAYER 2:
  Z = D̃^(-½) Ã D̃^(-½) · H¹ · W₂

Where:
  W₂  = Learnable weight matrix ∈ ℝ^(128 × 64)
  Z   = Final node embeddings ∈ ℝ^(659 × 64)
```

**Why Dropout + BatchNorm?** On a sparse graph, a basic GCN memorizes training edges and scores fake "99%" accuracy. BatchNorm prevents vanishing gradients, and 45% Dropout forcibly randomizes the network to generalize on completely hidden test edges.

#### The Decoder — Bilinear Matrix + MLP Neural Network

Unlike the basic dot-product (`z_u · z_v`), we use a **learned, asymmetric scoring function**:

```
BILINEAR SCORE:
  score_bilinear = z_u^T · W · z_v

Where:
  W = Learnable Bilinear matrix ∈ ℝ^(64 × 64)
  This allows the model to learn that, e.g.,
  "treatment" → "disorder" scores differently than "disorder" → "treatment"

MLP SCORE (Feature Fusion):
  concat_features = [z_u | z_v | z_u ⊙ z_v]    ← 192-dimensional input
  h₁ = ReLU( W_mlp1 · concat_features + b₁ )   ← 128 hidden units
  h₂ = ReLU( W_mlp2 · h₁ + b₂ )               ← 64 hidden units
  score_mlp = W_mlp3 · h₂ + b₃                 ← Scalar output

FINAL PREDICTION:
  score_final = score_bilinear + score_mlp
  P(link exists) = Sigmoid( score_final ) ∈ [0, 1]
```

**Why this matters**: The dot-product only asks *"are these vectors similar?"* The Bilinear+MLP Decoder asks *"in what specific ways, and through what complex interaction terms, do these two concepts predict an undiscovered connection?"*

#### Multi-Seed Training — Selecting the Optimal Model

```
FOR seed IN [42, 123, 456]:
    Initialize model with random seed
    Train for 400 epochs with ReduceLROnPlateau scheduler
    Evaluate on Validation Set (10% held-out edges)
    IF val_auc > best_so_far:
        SAVE model to gnn_model.pt

Final saved model: Best across 3 initializations
```

This guarantees we don't get accidentally stuck in a poor local minimum in the loss landscape.

### 📊 **Explainable 3D Dashboard**
- Real-time Plotly 3D graph rotation and zoom
- AI transparency panels showing:
  - Strict Test and Validation performance metrics
  - Complete Neural architecture details
  - Top AI research gap predictions (with actual Probability Scores)
  - Full end-to-end Methodology explanation

---

## 🚀 **Quick Start**

### Prerequisites
```bash
Python 3.11+
pip
virtualenv
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/srivardhan-kondu/The-Negative-Knowledge.git
cd The-Negative-Knowledge
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
python -m spacy download en_ner_bc5cdr_md
```

4. **Launch the Dashboard**
```bash
streamlit run scripts/visualize_credible_ai.py
```

Your browser will open showing the interactive 3D visualization and dashboard!

---

## 📖 **How It Works**

### End-to-End Pipeline

```
┌─────────────────┐
│ 1. Data Fetch   │  ──▶  fetch_papers_large.py (872 papers)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 2. NLP Extract  │  ──▶  extract_entities.py (46,000 concepts via scispaCy)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 3. Graph Build  │  ──▶  build_pyg_graph.py (Node2Vec + SciBERT embeddings)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 4. GNN Training │  ──▶  train_gnn.py (GCN Encoder + Bilinear MLP Decoder)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 5. Transparent UI│  ──▶  visualize_credible_ai.py (3D Interactive UI)
└─────────────────┘
```

---

## 📊 **Results**

### Model Performance

Unlike prototype models that overfit on their own training data to claim "99%" accuracy, this framework enforces a strict **80/10/10 Train/Validation/Test split**. 

The model must predict completely hidden, surgically removed associations.

| Metric | Value |
|--------|-------|
| **Validation ROC-AUC** | 76.8% |
| **Strict Test ROC-AUC** | **74.1%** |
| **Prediction Meaning** | Given a real missing medical link and a completely false link, the AI correctly identifies the real one ~74% of the time. |

### Top 5 Discovered Research Gaps

1. **Trauma-focused CBT ↔ Depression** (82.9% confidence)
2. **Loneliness ↔ Beck Anxiety Inventory** (82.6%)
3. **Advanced Treatments ↔ Bipolar Disorder** (81.6%)
4. **Monotherapy ↔ Anxiety** (81.4%)
5. **Depression ↔ Quality of Sleep** (80.5%)

> These predictions represent highly probable but under-researched connections strongly suggested by the geometry of the existing literature.

---

## 🛠️ **Project Structure**

```
The-Negative-Knowledge/
├── scripts/              # All pipeline scripts
│   ├── fetch_papers_large.py     # High-throughput data collection
│   ├── extract_entities.py       # NLP entity extraction
│   ├── classify_entities.py      # Category classification
│   ├── extract_relations.py      # Relation extraction
│   ├── build_graph.py            # Knowledge graph construction
│   ├── train_node2vec.py         # 128D Topology training
│   ├── train_semantic_embeddings.py # 768D SciBERT training
│   ├── build_pyg_graph.py        # 896D Feature Concatenation
│   ├── train_gnn.py              # Bilinear Decoder model training
│   └── visualize_credible_ai.py  # 3D visualization dashboard
├── data/                 # Data files (gitignored)
│   ├── mindgap.db               # SQLite database
│   ├── pyg_graph_splits.pt      # Train/Val/Test PyTorch splits
│   ├── gnn_model.pt             # Trained GCN + Decoder Dict
│   └── graph_credible_ai.html   # Final visualization UI output
├── config.yaml           # Domain configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## 🌟 **Advanced Features**

### Multi-Domain Support

The system supports **any medical domain** through configuration:

```yaml
# config.yaml
domains:
  mental_health: ...
  diabetes: ...
  cancer: ...
```

---

## 🎓 **Tech Stack**

### AI/ML
- **PyTorch** + **PyTorch Geometric** - Deep learning, Data Splitting
- **Transformers (HuggingFace)** - `SciBERT` embeddings
- **Node2Vec** - DeepWalk graph topology embeddings
- **NetworkX** - Complex graph manipulation

### NLP
- **spaCy** - Core NLP pipeline
- **scispaCy** - Highly specialized Biomedical NER
- **en_core_sci_sm** - Broad Scientific corpus
- **en_ner_bc5cdr_md** - Disease/Chemical corpus

### Data & UI
- **SQLite** - High-speed Local DB
- **Plotly** - WebGL 3D Network graphs
- **Streamlit** - Python UI Framework

---

## 🤝 **Contributing**

Contributions are welcome! Please open an issue or submit a Pull Request.

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 **Contact**

**Srivardhan Kondu**
- GitHub: [@srivardhan-kondu](https://github.com/srivardhan-kondu)
- Project: [The Negative Knowledge](https://github.com/srivardhan-kondu/The-Negative-Knowledge)

---

<p align="center">
  <strong>Built with ❤️ for advancing scientific research</strong>
  <br>
  <sub>Strict Evaluation • Fully Transparent • Open Source</sub>
</p>
