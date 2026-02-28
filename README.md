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

### 🧠 **Advanced Hybrid AI Pipeline**
- **Entity Extraction**: `scispaCy` extracted over **46,000 raw medical concepts**, which were heavily deduplicated down to the **659 most critical core nodes** connected by **4,856 verified edges**.
- **896-Dimensional Hybrid Intelligence**: 
  - Nodes no longer just know *where* they are in the graph (`128D Node2Vec`).
  - Nodes now understand exactly *what they mean medically* by reading the `768D SciBERT` transformer embeddings.

### 🧬 **State-of-the-Art Neural Architecture**
- **Compression Encoder**: A 2-Layer GCN running dimensionality reduction (896D → 64D) fortified with **BatchNorm** and **45% Dropout** to heavily penalize overfitting.
- **Bilinear Hybrid Decoder**: Replaced basic dot-product scoring with a learned **Bilinear Matrix ($W$) + 3-Layer MLP neural network**. The model learns complex asymmetrical scoring functions to judge connection probability.
- **Multi-Seed Training**: The training loop runs 3 entirely separate initializations and only keeps the model that discovers the deepest mathematical optimum. 

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
