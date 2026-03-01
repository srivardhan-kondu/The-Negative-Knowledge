# 🧠 The Negative Knowledge

### AI-Powered Research Gap Discovery using Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ROC-AUC](https://img.shields.io/badge/Test_ROC--AUC-97.22%25-brightgreen.svg)]()
[![Mobile](https://img.shields.io/badge/UI-Mobile_Responsive-blueviolet.svg)]()

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

### 📊 **Client-Server Explainable Dashboard — 4-Tab Layout**
- **Backend**: Flask REST API (`server.py`) serving predictions, graph data, and metrics in real-time.
- **Frontend**: Premium HTML/CSS/JS Single Page Application — **fully mobile responsive**.
- 4 dedicated tabs:
  - 🔍 **Search** — type any concept to find the GNN-predicted missing connections
  - 🌐 **3D Graph** — full-viewport interactive 3D graph (all 659 nodes), free-orbit drag modes, auto-rotate
  - 🔴 **Top 20 Gaps** — dedicated page listing all 20 highest-confidence AI-predicted research gaps
  - 📊 **Model Metrics** — test ROC-AUC, graph stats, architecture details, pie chart
- **Batch-scored predictions**: all 15,000 candidate pairs scored in a single PyTorch forward pass (~1 sec)
- **Mobile responsive**: collapsible sidebar overlay, scrollable tab nav, 60vh graph on phones, iOS safe-area insets

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

4. **Launch the Application**
```bash
bash run.sh
```

Your browser will automatically open `http://localhost:5050` showing the interactive 3D visualization and dashboard!

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
│ 5. API Backend  │  ──▶  server.py (Flask REST API serving predictions)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 6. Frontend SPA │  ──▶  frontend/index.html (4-tab Mobile-Responsive UI)
└─────────────────┘
```

---

## 📊 **Results**

### Model Performance

Unlike earlier prototypes that evaluated on training edges (inflated ~99% accuracy), this framework enforces **strict evaluation on held-out test edges** — edges the model has never seen during training.

| Metric | Value |
|--------|-------|
| **Test ROC-AUC (held-out edges)** | **97.22%** |
| **Prediction Meaning** | Given a real hidden connection and a fake one, the AI correctly identifies the real one ~97% of the time on edges it never trained on. |
| **Scoring Speed** | ~1 second — all 15,000 pairs batch-scored in a single forward pass |

### Top 5 Discovered Research Gaps

1. **Mindfulness ↔ Mindfulness Teachers** (91.9% confidence)
2. **Antidepressants ↔ Treatment** (85.9%)
3. **HADS-Anxiety Subscale ↔ Depression** (85.6%)
4. **Late Adulthood ↔ Depression** (85.6%)
5. **DBT ↔ Depression** (84.5%)

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
│   ├── visualize_credible_ai.py  # Old static HTML generator
│   └── streamlit_app.py          # Legacy Streamlit prototype
├── frontend/             # Single Page Application UI
│   ├── index.html               # Main dashboard layout
│   ├── styles.css               # Premium dark theme styling
│   └── app.js                   # API client and Plotly 3D rendering
├── data/                 # Data files (gitignored)
│   ├── mindgap.db               # SQLite database
│   ├── pyg_graph_splits.pt      # Train/Val/Test PyTorch splits
│   └── gnn_model.pt             # Trained GCN + Decoder Dict
├── server.py             # Flask REST API Backend
├── run.sh                # Super-convenience launcher script
├── demo.sh               # Alias to run.sh for demos
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

### Data & Web
- **SQLite** - High-speed Local DB
- **Flask** - REST API Backend serving the PyTorch model
- **HTML/CSS/JS** - Lightweight, high-performance web frontend
- **Plotly.js** - WebGL 3D Network graphs

---

## 🤝 **Contributing**

Contributions are welcome! Please open an issue or submit a Pull Request.

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 **Project Review Q&A Guide**

> **For the team: This section covers every question your guide or panel could ask you. Read this before your review.**

---

### 🔷 SECTION 1: Problem & Motivation

**Q1. What is "Negative Knowledge" and why is it important?**
> Negative Knowledge refers to research connections that *should* exist but have never been explored. Think of it as the "unknown unknowns" of science. For example, no study has yet investigated the relationship between *Loneliness* and the *Beck Anxiety Inventory* scale — even though both concepts appear frequently in mental health literature. Identifying these gaps helps redirect scientific effort toward truly impactful, unexplored territory.

**Q2. Why can't researchers just do a keyword search to find gaps?**
> A keyword search can find what *has* been written. It cannot tell you what *hasn't* been written. Our GNN learns the geometric structure of the entire knowledge graph — the topology of thousands of connections — and predicts which pairs of nodes should logically be connected but aren't. Keyword search is local; our model is global.

**Q3. Why is mental health the chosen domain?**
> Mental health was chosen because: (1) it has a large volume of accessible literature on Semantic Scholar and PubMed, (2) the entities (disorders, therapies, risk factors) are well-defined and amenable to NLP extraction, and (3) research gaps in this domain have real clinical significance. The framework is fully domain-agnostic and can be configured to work for Diabetes or Cancer with a single config change.

---

### 🔷 SECTION 2: Data Collection & NLP

**Q4. Where does the data come from?**
> We use two sources:
> - **Semantic Scholar** — A free, open academic API from the Allen Institute for AI. We query it using 20 different mental health search terms (depression, anxiety, PTSD, CBT, etc.) to maximize coverage.
> - **PubMed (NIH Entrez)** — The National Institutes of Health official research database. We use a 2-step API: `esearch` to get IDs, then `efetch` to retrieve abstracts.
>
> In total we collected **872 research papers**.

**Q5. How do you extract medical concepts from raw text?**
> We use **two scispaCy NER models** running simultaneously on every abstract:
> 1. `en_core_sci_sm` — Trained on broad biomedical corpora, extracts general scientific terms.
> 2. `en_ner_bc5cdr_md` — Trained specifically on the BioCreative V CDR dataset (diseases + chemicals). 
>
> Every entity is lowercased and deduplicated. We extracted over **46,000 raw entities** and deduplicated them down to **659 core nodes**.

**Q6. How do you decide which entities are valid and which are noise?**
> We use a **5-category keyword matching filter**. An entity is only kept if it belongs to one of: `Disorder`, `Therapy`, `Risk Factor`, `Outcome`, or `Population`. Entities outside these categories (generic nouns, verbs, place names) are discarded.

**Q7. How are relationships between entities established?**
> Through **sentence-level co-occurrence**. If two categorized entities appear in the same sentence within an abstract, they are connected with a `related_to` edge in the knowledge graph. spaCy's `en_core_web_sm` model handles sentence boundary detection.

---

### 🔷 SECTION 3: Knowledge Graph

**Q8. What kind of graph is this?**
> An **undirected, weighted NetworkX graph** stored in Python. After deduplication and filtering, our final graph has **659 nodes** (medical concepts) and **4,856 edges** (co-occurrence relationships). The graph is then converted to a **PyTorch Geometric `Data` object** for GNN training.

**Q9. What is the sparsity problem, and how did you address it?**
> Most nodes in the graph are connected to only a few other nodes. This makes it extremely hard for a GNN to learn useful patterns, because there isn't enough topological signal. We addressed this by:
> 1. **Increasing the dataset** from 650 to 872 papers to create a denser graph.
> 2. **Adding semantic features (SciBERT)** so the AI doesn't rely *only* on graph topology and can use medical meaning.
> 3. **Using aggressive Dropout** in the encoder to prevent memorization.

---

### 🔷 SECTION 4: Node Embeddings (896D Hybrid)

**Q10. What is Node2Vec and why 128 dimensions?**
> Node2Vec is a graph embedding method that performs **biased random walks** on the graph. Each walk is a sequence of nodes, like a sentence. Word2Vec's Skip-gram model is then trained on these sequences, outputting a dense vector for every node. We chose 128 dimensions as a balance between information capacity and computational cost. Key parameters:
> - `p=1.0` (return probability), `q=0.5` (exploration bias toward BFS)
> - `K=200` walks per node, `L=20` steps per walk

**Q11. What is SciBERT and why is it important here?**
> SciBERT is a **BERT-based transformer model** pre-trained specifically on **1.14 million academic papers** from Semantic Scholar. For every medical concept (e.g., "dialectical behavioral therapy"), we pass its name as text through SciBERT. The output `[CLS]` token from the final transformer layer gives us a **768-dimensional vector** that encodes the deep semantic meaning of the concept — even if the node is isolated in the graph with no connections.

**Q12. Why concatenate instead of average the two embeddings?**
> **Averaging destroys information**. If Node2Vec says "CBT is central in the graph" (high-norm topology vector) and SciBERT says "CBT is a structured cognitive intervention" (specific semantic direction), an average blends them into a meaningless middle. Concatenation `[128D | 768D]` keeps both complete signals separate and lets the GNN Encoder decide how to weigh them.

---

### 🔷 SECTION 5: GNN Model Architecture

**Q13. What is a Graph Convolutional Network (GCN)?**
> A GCN is a neural network designed to operate on graph-structured data. Each GCN layer performs **message passing** — every node collects feature vectors from its neighbors, aggregates them, and updates its own representation. The formula is:
> ```
> H^(l+1) = σ( D̃^(-½) Ã D̃^(-½) H^(l) W^(l) )
> ```
> Where `Ã` is the adjacency matrix with added self-loops, `D̃` is the normalized degree matrix, and `W` is the trainable weight matrix. After 2 GCN layers, each node's 896D vector is compressed to 64D, encoding information from its 2-hop neighborhood.

**Q14. Why specifically 2 GCN layers and not more?**
> Adding more GCN layers causes the **over-smoothing problem** — node representations become too similar to each other and lose their individual identity. 2 hops is sufficient to capture both direct neighbors (1-hop) and neighbors-of-neighbors (2-hop), which is the most informative neighborhood for link prediction in knowledge graphs.

**Q15. What is BatchNorm and why did you add it?**
> BatchNorm (Batch Normalization) normalizes the activations of each layer to have zero mean and unit variance across a mini-batch. It **stabilizes training**, prevents vanishing/exploding gradients, and allows higher learning rates. In our sparse graph, raw GCN activations can have wildly different scales, which makes optimization difficult.

**Q16. Why 45% Dropout?**
> Dropout randomly *switches off* 45% of neurons during each training step, forcing the network to learn **robust, distributed representations** instead of relying on a small set of neurons. This heavily penalizes memorization and forces the model to generalize to the hidden test edges. Without dropout, our model would achieve "99%" on training data and fail completely on unseen test edges.

---

### 🔷 SECTION 6: Link Prediction & Decoder

**Q17. What is link prediction?**
> Link prediction is the task of predicting whether an edge should exist between two nodes that are *not currently connected*. In our context: given two medical concepts that have never been co-studied, what is the probability that they represent a valuable research gap?

**Q18. Why is the dot-product decoder insufficient?**
> A dot-product decoder simply computes `score = z_u · z_v` — it measures cosine-style similarity. This assumes that "nodes with similar embedding vectors should be connected," which is a very linear, symmetric assumption. Medical relationships are rarely this simple. Drug A might affect Disease B differently than Disease B affects Drug A. Our decoder is fundamentally more powerful: `score = z_u^T · W · z_v + MLP([z_u | z_v | z_u ⊙ z_v])`.

**Q19. Explain the Bilinear Matrix in the decoder.**
> The Bilinear Matrix `W ∈ ℝ^(64×64)` learns an **asymmetric interaction** between any two node embeddings. Where a dot product just measures alignment, the bilinear term `z_u^T W z_v` can learn to score asymmetric relationships: "therapy concepts should connect to disorder concepts" can have a different weight pattern than "disorder concepts connecting to each other."

**Q20. Explain the MLP Fusion part of the decoder.**
> The MLP takes the concatenation of three vectors: `[z_u | z_v | z_u ⊙ z_v]` — the two node embeddings plus their **element-wise product** (which captures feature-level interactions). This 192D vector passes through:
> - Layer 1: `192 → 128` (ReLU)  
> - Layer 2: `128 → 64` (ReLU)  
> - Layer 3: `64 → 1` (Scalar score)
>
> The MLP learns non-linear, complex interaction patterns the bilinear term alone cannot capture.

---

### 🔷 SECTION 7: Training & Evaluation

**Q21. How is the training data split?**
> We use a strict **80 / 10 / 10 split**:
> - **Training (80%)**: The model sees this and updates its weights.
> - **Validation (10%)**: Used during training to monitor performance and trigger early stopping.
> - **Test (10%)**: Completely **hidden** from the model during training. Only used for final reporting.
>
> Negative samples (non-edges) are randomly sampled at equal count to the positive samples per epoch.

**Q22. What does 74.1% Test ROC-AUC actually mean?**
> It means: if you randomly give the model one **real** hidden research gap (a removed edge) and one **fake** connection (a randomly sampled non-edge), the model correctly identifies the real one **74.1% of the time**. This is a rigorous, threshold-independent metric. For comparison, a random classifier scores exactly 50%.

**Q23. Why is 74.1% a good result if "99.76%" was achieved before?**
> The previous 99.76% was **not a real result**. The model was trained and tested on the exact same data, so it simply memorized the training edges. Our 74.1% is measured on **completely hidden test edges that the model never saw** during training. For biomedical knowledge discovery on a sparse graph of 872 papers, 74.1% is a scientifically honest and strong result.

**Q24. What is Multi-Seed training?**
> We run the full training loop 3 separate times with different random seeds (`42`, `123`, `456`). Each run initializes the model weights differently. After all 3 runs, we keep only the model that achieved the **highest validation AUC**. This protects against getting unlucky with a bad weight initialization.

---

### 🔷 SECTION 8: Explainability (XAI)

**Q25. How is this project "explainable AI"?**
> The visualization dashboard explicitly shows:
> 1. The **exact model architecture** (layer sizes, decoder type)
> 2. The **training parameters** (epochs, learning rate, dropout)
> 3. The **validated performance metric** (Test ROC-AUC: 74.1%)
> 4. **Each prediction's probability score** (not just a black-box "yes/no")
> 5. A **step-by-step methodology panel** — users can see exactly how predictions were generated
> 6. **Responsible AI disclaimers** — predictions are described as suggestions for further investigation, not verified facts

---

### 🔷 SECTION 9: Limitations & Future Work

**Q26. What are the current limitations?**
> 1. **Graph sparsity** — 872 papers is still small for a knowledge graph task
> 2. **Negation blindness** — "CBT is *not* effective for PTSD" still creates a CBT-PTSD edge
> 3. **English only** — Ignores non-English literature
> 4. **Offline pipeline** — No real-time updates when new papers are published
> 5. **Single domain tested** — Only Mental Health is fully demonstrated

**Q27. How would you improve accuracy further?**
> 1. Expand to 5,000–10,000+ papers (requires data sources without strict API rate limits)
> 2. Use **negation detection** (`negspacy`) to avoid false-positive relations
> 3. Replace co-occurrence with **transformer-based relation extraction** (e.g., REBEL)
> 4. Add **GAT (Graph Attention Networks)** to learn importance weights for neighbors
> 5. Integrate with large pre-existing biomedical KGs like **UMLS or DisGeNET**

---

> 💡 **Team Tip:** If asked anything unexpected, anchor back to the core insight — *"Our system identifies research connections that the entire published literature has missed, and it tells you exactly why it thinks they're important."*

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

