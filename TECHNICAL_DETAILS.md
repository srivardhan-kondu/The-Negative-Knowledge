# 📖 Technical Details — Negative Knowledge

## An Explainable GNN-Based Framework for Negative Knowledge Discovery Using Scientific Knowledge Graphs

End-to-end technical documentation of the system architecture, data pipeline, model design, and explainability framework.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![ROC-AUC](https://img.shields.io/badge/Test_ROC--AUC-97.22%25-brightgreen.svg)]()
[![Mobile](https://img.shields.io/badge/UI-Mobile_Responsive-blueviolet.svg)]()

---

## 🎯 What is Negative Knowledge?

**Negative Knowledge** represents the *unknown unknowns* in scientific research — connections that should exist but haven't been studied yet. While traditional literature reviews tell us what has been studied, Negative Knowledge reveals what has been *missed*.

> **The Problem:** Over 2.5 million research papers are published every year. No researcher can read all of them. This means promising connections between concepts studied in different sub-fields go unnoticed for years or decades.

**Our Solution:** Using **Graph Neural Networks (GNNs)** combined with **deep biomedical language models (SciBERT)**, we construct knowledge graphs from research papers and predict which connections are missing. The system explains every prediction through gradient-based GNN attribution.

---

## 📖 End-to-End Pipeline

```
┌────────────────────┐
│  1. Data Fetch     │  → fetch_papers_large.py
│                    │    872 papers from Semantic Scholar + PubMed
└────────┬───────────┘
         ▼
┌────────────────────┐
│  2. NLP Extract    │  → extract_entities.py
│                    │    46,000 raw entities via 2 scispaCy models
└────────┬───────────┘
         ▼
┌────────────────────┐
│  3. Classify       │  → classify_entities.py + domain_config.py
│                    │    5-category keyword classification → 659 nodes
└────────┬───────────┘
         ▼
┌────────────────────┐
│  4. Relations      │  → extract_relations.py
│                    │    Sentence-level co-occurrence → 4,856 edges
└────────┬───────────┘
         ▼
┌────────────────────┐
│  5. Graph Build    │  → build_graph.py → NetworkX pickle
│                    │    train_node2vec.py → 64D topology vectors
│                    │    train_semantic_embeddings.py → 768D SciBERT
│                    │    build_pyg_graph.py → 896D PyG Data object
└────────┬───────────┘
         ▼
┌────────────────────┐
│  6. GNN Training   │  → train_gnn.py
│                    │    GCNEncoder + HybridDecoder, 3 seeds, 600 epochs
└────────┬───────────┘
         ▼
┌────────────────────┐
│  7. API Backend    │  → server.py
│                    │    Flask REST API on port 5050
└────────┬───────────┘
         ▼
┌────────────────────┐
│  8. Frontend SPA   │  → frontend/ (index.html + styles.css + app.js)
│                    │    5-tab dashboard + 3D graph + "Why?" modals
└────────────────────┘
```

---

## 📦 Data Collection Pipeline

| Source | API | Papers Queried | Rate Limiting | Fields Retrieved |
|--------|-----|----------------|---------------|-----------------|
| **Semantic Scholar** | `graph/v1/paper/search` | 25 pages × 100/page × 20 terms = 50,000 candidates | 1.2s between pages; 60s backoff on HTTP 429 | title, abstract, year, authors (first 5), venue |
| **PubMed (NIH Entrez)** | `esearch.fcgi` → `efetch.fcgi` | 200 results/term × 20 terms = 4,000 candidates | 0.4s between papers; 1s between terms | title, abstract, year, authors (first 5) |

### Search Strategy

20 carefully chosen mental health search terms:

```
depression mental health, anxiety disorder treatment, PTSD therapy,
cognitive behavioral therapy, mindfulness meditation, bipolar disorder,
schizophrenia diagnosis, suicide prevention, substance abuse treatment,
eating disorder, social anxiety, OCD treatment, panic disorder,
trauma and mental health, psychotherapy outcomes, antidepressant efficacy,
sleep disorders mental health, stress management, adolescent mental health,
group therapy effectiveness
```

### Deduplication and Quality Control

- Papers stored with unique `paper_id` keys; `INSERT OR IGNORE` prevents duplicates
- Abstracts shorter than 100 characters are discarded
- Author lists truncated to first 5 authors
- **Final dataset: 872 unique mental health research papers**

---

## 🔬 NLP Entity Extraction Pipeline

### Stage 1: Biomedical Named Entity Recognition (NER)

| Model | Training Data | What It Extracts | Strength |
|-------|--------------|------------------|----------|
| `en_core_sci_sm` | Broad biomedical corpora | General scientific terms | High recall |
| `en_ner_bc5cdr_md` | BioCreative V CDR dataset | Diseases and chemicals | High precision |

### Stage 2: Synonym Normalization

| Raw Entity | Normalized Form |
|-----------|----------------|
| major depressive disorder | depression |
| depressive symptoms | depression |
| ptsd | post-traumatic stress disorder |
| cbt | cognitive behavioral therapy |
| dbt | dialectical behavior therapy |
| severe anxiety | anxiety |

### Stage 3: Entity Classification

| Category | Example Keywords | Purpose |
|----------|-----------------|---------|
| **Disorder** | depression, anxiety, ptsd, bipolar, panic, ocd | Medical conditions |
| **Therapy** | therapy, cbt, dbt, treatment, counseling, mindfulness | Interventions |
| **Risk Factor** | trauma, abuse, stress, insomnia, loneliness | Risk variables |
| **Outcome** | suicide, relapse, recovery, quality of life | Measured results |
| **Population** | adolescent, child, women, men, elderly, veteran | Demographics |

### Stage 4: Relation Extraction via Sentence-Level Co-occurrence

```
For each paper abstract:
  Parse into sentences using spaCy
  For each sentence:
    Find all categorized entities (substring match)
    Optional: Run negation detection (NegEx via negspacy)
    Exclude negated entities
    Create pairwise "related_to" edges between all remaining entity pairs
```

### Output Statistics

| Metric | Value |
|--------|-------|
| Raw entities extracted | ~46,000 |
| After deduplication + normalization | 659 unique concepts |
| Total relations extracted | 4,856 unique edges |
| Relation type | `related_to` (sentence co-occurrence) |

---

## 🕸️ Knowledge Graph

| Property | Value | Significance |
|----------|-------|-------------|
| **Type** | Undirected, weighted | Co-occurrence is symmetric |
| **Nodes** | 659 | Unique medical concepts |
| **Edges** | 4,856 | Co-occurrence relationships |
| **Density** | ~2.24% | Extremely sparse graph |
| **Average Degree** | ~14.7 | Each concept ≈ 15 connections |
| **Storage** | NetworkX → PyTorch Geometric | GPU-accelerated GNN training |

### The Sparsity Challenge

With only 2.24% density, we addressed sparsity through:
- **Increased dataset scale**: Expanded from 650 to 872 papers
- **Semantic features (SciBERT)**: 768-D embeddings provide information even for weakly-connected nodes
- **Aggressive regularization**: 45% Dropout + BatchNorm

---

## 🧠 Hybrid 896-Dimensional Feature Vector

> **Core Innovation:** Every node is represented by a two-part 896D vector encoding both graph structure AND medical semantics.

### Part 1: Node2Vec — 128D (Topology)

- Biased random walks (p=1.0, q=0.5)
- 200 walks × 20 steps per node
- Word2Vec Skip-gram → 64D vectors
- Zero-padded to 128D
- Captures: *"who are your neighbors in the graph?"*

### Part 2: SciBERT — 768D (Semantics)

- Pre-trained on 1.14M scientific papers
- 12 Transformer layers, 110M parameters
- Custom 31,116-token scientific vocabulary
- [CLS] token extraction for sentence embedding
- Captures: *"what do you actually mean medically?"*

### Final Combination

```
z_node = CONCATENATE(z_topo, z_sem)
       = [128D Node2Vec (padded) | 768D SciBERT]
       = 896-Dimensional Final Feature Vector
```

**Why concatenation, not averaging?** Averaging destroys information. Concatenation preserves both signals completely and lets the downstream GCN learn how to weigh them differently per prediction.

---

## 🧬 GNN Architecture

### GCN Encoder

```
INPUT:   X ∈ ℝ^(659 × 896)   — Raw 896-D hybrid feature matrix

Step 0 — Input Normalization:
  BatchNorm1d(896)              — Zero mean, unit variance

Step 1 — Linear Projection:
  Linear(896 → 128) + ReLU     — Dimensionality reduction

Step 2 — First GCN Layer:
  GCNConv(128 → 128)           — 1st graph convolution
  BatchNorm + ReLU + Dropout(0.45)

Step 3 — Second GCN Layer:
  GCNConv(128 → 64)            — 2nd graph convolution
  BatchNorm

OUTPUT:  Z ∈ ℝ^(659 × 64)    — Learned 64-D node embeddings
```

### GCN Message-Passing Formula

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

### Hybrid Decoder (Bilinear + MLP)

**Branch 1 — Bilinear Scoring:**
```
score_bilinear = zᵤᵀ · W · zᵥ + b    (W ∈ ℝ^64×64 = 4,096 parameters)
```

**Branch 2 — MLP Fusion:**
```
INPUT:  [zᵤ | zᵥ | zᵤ ⊙ zᵥ]    — 192-dimensional vector
Layer 1: 192 → 32 (ReLU + Dropout 0.4)
Layer 2: 32 → 1 (scalar output)
```

**Final Score:**
```
P(link exists) = Sigmoid(score_bilinear + score_mlp) ∈ [0, 1]
```

### Why 2 GCN Layers?

| Layers | Reach | Risk |
|--------|-------|------|
| 1 | Direct neighbors only | Too local |
| **2** | **2-hop neighborhood** | **Sweet spot ✓** |
| 3+ | Nearly entire graph | Over-smoothing |

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Input BatchNorm | 1,792 |
| Linear Projection (896→128) | 114,816 |
| GCNConv Layer 1 (128→128) | 16,512 |
| BatchNorm 1 | 256 |
| GCNConv Layer 2 (128→64) | 8,256 |
| BatchNorm 2 | 128 |
| Bilinear Decoder (64×64 + bias) | 4,097 |
| MLP (192→32→1) | 6,177 |
| **Total** | **~152,034** |

---

## 🏋️ Training Pipeline

### Data Splitting

| Split | Proportion | Edges (~) | Purpose |
|-------|-----------|-----------|---------|
| Training | 80% | ~3,885 | Model learns weights |
| Validation | 10% | ~485 | Early stopping |
| Test | 10% | ~486 | Final evaluation (never seen) |

### Negative Sampling

| Type | Proportion | Purpose |
|------|-----------|---------|
| Random negatives | 75% | Teaches "most random pairs shouldn't connect" |
| Hard negatives (2-hop) | 25% | Forces fine-grained discrimination |
| Label smoothing | 0.1 | Prevents overconfidence |

### Optimizer and Schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive LR per parameter |
| Initial LR | 0.003 | Aggressive start |
| Weight Decay | 5×10⁻⁴ | L2 regularization |
| Scheduler | CosineAnnealingWarmRestarts | Escape local minima via periodic restarts |
| T₀ | 150 epochs | Restart period |
| η_min | 5×10⁻⁶ | LR floor |
| Gradient Clipping | max_norm = 1.0 | Prevent exploding gradients |
| Early Stopping | 80 epochs patience | Stop when val AUC plateaus |

### Multi-Seed Training

```
FOR seed IN [42, 123, 999]:
    Initialize model with fresh random weights
    Train for up to 600 epochs with early stopping
    Evaluate on Validation Set
    IF val_auc > best_so_far: SAVE model
```

---

## 🔬 Explainable AI — Gradient-Based GNN Attribution

Every prediction is **fully explainable** using gradient-based attribution — the GNN equivalent of **Grad-CAM** (Selvaraju et al., ICCV 2017). This is model-intrinsic, computed in <200ms, and decomposes into topology vs. semantics contributions.

### How It Works

```
Step 1: Enable gradients on 896D input features
Step 2: Forward pass through GCN Encoder → 64D embeddings
Step 3: Compute link score via HybridDecoder
Step 4: Backpropagate gradients to input features
Step 5: importance[d] = |gradient[d]| × |feature[d]|
Step 6: Split — dims 0-127 = topology, dims 128-895 = semantics
```

### 8-Panel Explanation Dashboard

| Panel | Content |
|-------|---------|
| **1. Confidence Gauge** | Calibrated score zones: Weak (50-60%), Moderate (60-70%), Strong (70-80%), Very Strong (80%+) |
| **2. Feature Attribution** | Topology vs. Semantics percentage split via gradient backpropagation |
| **3. Top Feature Dims** | 10 most activated dimensions in 896D vector, color-coded by source |
| **4. Common Neighbors** | Bridge concepts connected to both nodes — structural evidence |
| **5. Shortest Path** | How the literature already connects two concepts indirectly |
| **6. Embedding Similarity** | Cosine similarity of 64D GNN-learned representations |
| **7. Influential Neighbors** | Existing neighbors that "pull" the prediction toward a positive link |
| **8. Paper Evidence** | Actual research papers from SQLite — data provenance |

### Topology vs Semantics Interpretation

- **High topology %** → Prediction driven by graph structure (neighborhood similarity)
- **High semantics %** → Prediction driven by medical meaning (SciBERT language understanding)
- **Balanced** → Both structure and meaning contribute (strongest predictions)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test ROC-AUC (held-out edges)** | **97.22%** |
| Prediction Meaning | Given a real hidden connection and a fake one, the AI correctly identifies the real one ~97% of the time on edges it never trained on |
| Scoring Speed | ~1 second — all 15,000 pairs batch-scored |

### Top 5 Discovered Research Gaps

| # | Prediction | Confidence |
|---|-----------|------------|
| 1 | Mindfulness ↔ Mindfulness Teachers | 91.9% |
| 2 | Antidepressants ↔ Treatment | 85.9% |
| 3 | HADS-Anxiety Subscale ↔ Depression | 85.6% |
| 4 | Late Adulthood ↔ Depression | 85.6% |
| 5 | DBT ↔ Depression | 84.5% |

---

## 🗄️ Database Architecture

All data stored in **SQLite** — zero-configuration, portable, ACID-compliant.

```sql
CREATE TABLE papers (
    paper_id  TEXT PRIMARY KEY,
    title     TEXT,
    abstract  TEXT,
    year      INTEGER,
    authors   TEXT,
    venue     TEXT,
    source    TEXT
);

CREATE TABLE entities (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id  TEXT,
    entity    TEXT,
    type      TEXT,
    source    TEXT,
    category  TEXT,
    UNIQUE(paper_id, entity, type)
);

CREATE TABLE relations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id  TEXT,
    head      TEXT,
    relation  TEXT,
    tail      TEXT,
    UNIQUE(paper_id, head, relation, tail)
);
```

---

## 🌐 REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Server status & model loaded |
| `/api/metrics` | GET | ROC-AUC, graph stats, architecture |
| `/api/predictions` | GET | Top-K globally predicted research gaps |
| `/api/search` | POST | Search by concept, return scored gaps |
| `/api/graph_data` | GET | Full 3D graph layout for Plotly |
| `/api/explain` | POST | 8-panel XAI explanation for a prediction |
| `/api/node_profile` | GET | Node category, degree, neighbors |

---

## 📊 Client-Server Dashboard — 5-Tab Layout

| Tab | Icon | Purpose | Data Source |
|-----|------|---------|-------------|
| **Search** | 🔍 | Type any concept to find GNN-predicted missing connections | `POST /api/search` |
| **3D Graph** | 🌐 | Full-viewport interactive 3D graph (all 659 nodes) | `GET /api/graph_data` |
| **Top 20 Gaps** | 🔴 | Ranked list of highest-confidence predicted research gaps | `GET /api/predictions` |
| **Model Metrics** | 📊 | Test ROC-AUC, graph stats, architecture details, pie chart | `GET /api/metrics` |
| **About** | 📖 | Complete technical documentation of the entire system | Static HTML |

### 3D Graph Visualization

- **Layout**: Fruchterman-Reingold force-directed (3D, spring constant k=2.0, 200 iterations)
- **Node sizing**: `15 + (degree / max_degree) × 30`
- **3 traces**: Known edges (gray), predicted gaps (red dashed), nodes (colored spheres)
- **Interactions**: Free-orbit rotation, auto-rotate, pan, zoom, drag mode switching

---

## 🎓 Tech Stack

### AI / ML

| Library | Purpose |
|---------|---------|
| PyTorch 2.0+ | Deep learning framework, GCN encoder, HybridDecoder |
| PyTorch Geometric | Graph neural network layers (GCNConv), link splitting |
| SciBERT (HuggingFace) | 768-D biomedical language embeddings |
| SentenceTransformers | Efficient [CLS] token extraction |
| Node2Vec | Biased random walks + Word2Vec for topology embeddings |
| NetworkX | Graph construction, shortest paths, neighbor queries |
| scikit-learn | Baseline models, GridSearchCV, ROC-AUC |
| XGBoost | Gradient boosting baseline |

### NLP

| Library | Purpose |
|---------|---------|
| spaCy | Core NLP pipeline (tokenization, sentence splitting) |
| scispaCy | Biomedical NLP extension |
| en_core_sci_sm | Broad scientific NER model |
| en_ner_bc5cdr_md | Disease/Chemical NER |
| en_core_web_sm | Sentence boundary detection |
| negspacy (optional) | Negation detection (NegEx) |

### Data & Web

| Technology | Purpose |
|-----------|---------|
| SQLite | Zero-config local database |
| Flask + Flask-CORS | REST API backend |
| HTML/CSS/JS | Vanilla SPA — no framework dependencies |
| Plotly.js | WebGL-accelerated 3D graph visualization |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/srivardhan-kondu/The-Negative-Knowledge.git
cd The-Negative-Knowledge

# Environment
python -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
python -m spacy download en_ner_bc5cdr_md

# Launch
bash run.sh
```

Browser opens at `http://localhost:5050` with the interactive dashboard.

---

## 📚 References

| Paper | Authors | Venue |
|-------|---------|-------|
| Semi-Supervised Classification with Graph Convolutional Networks | Kipf & Welling | ICLR 2017 |
| node2vec: Scalable Feature Learning for Networks | Grover & Leskovec | KDD 2016 |
| SciBERT: A Pretrained Language Model for Scientific Text | Beltagy et al. | EMNLP 2019 |
| Grad-CAM: Visual Explanations from Deep Networks | Selvaraju et al. | ICCV 2017 |
| GNNExplainer: Generating Explanations for Graph Neural Networks | Ying et al. | NeurIPS 2019 |
| Explainability Methods for Graph Convolutional Neural Networks | Pope et al. | CVPR 2019 |
| The Link Prediction Problem for Social Networks | Liben-Nowell & Kleinberg | 2003 |

---

*This document is auto-generated from the project's About tab and README.*
