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

**Negative Knowledge** represents the *unknown unknowns* in scientific research — **connections that should exist but haven't been studied yet**. While traditional literature reviews tell us what has been studied, Negative Knowledge reveals what has been *missed*. This project uses cutting-edge AI to discover these hidden research opportunities in medical science.

### The Problem

Scientific knowledge is growing exponentially. Over **2.5 million research papers** are published every year across all fields. No single researcher — or even an entire department — can read all existing literature. This means:

- 📚 **Information overload**: Researchers can only read a tiny fraction of papers in their own field
- 🔍 **Siloed knowledge**: Connections between concepts studied in different sub-fields go unnoticed
- ❓ **Missed opportunities**: Promising research directions remain unexplored for years or decades
- 💡 **Redundant research**: Scientists unknowingly duplicate work because they couldn't find existing connections

A keyword search can tell you what *has* been written. It **cannot** tell you what *hasn't* been written. Our system fills this gap — it learns the geometric structure of the *entire* knowledge graph and predicts which connections *should logically exist* but don't.

### Our Solution

Using **Graph Neural Networks (GNNs)** combined with **deep biomedical language models (SciBERT)**, we construct vast knowledge graphs from raw research papers and algorithmically predict which research connections are missing with a robust **74.1% true unseen test accuracy**.

The system doesn't just list predictions — it **explains every prediction** through gradient-based GNN attribution, showing exactly *why* the model believes a connection is missing and decomposing its reasoning into topology (graph structure) vs. semantics (medical meaning).

---

## 📦 **Data Collection Pipeline**

The quality and scale of the input data directly determines the quality of the knowledge graph and all downstream predictions. We built a **high-throughput, multi-source data collection pipeline** that maximizes coverage while respecting API rate limits.

### Data Sources

| Source | API | Papers Queried | Rate Limiting | Fields Retrieved |
|--------|-----|----------------|---------------|-----------------|
| **Semantic Scholar** | `graph/v1/paper/search` | 25 pages × 100/page × 20 terms = 50,000 candidates | 1.2s between pages; 60s backoff on HTTP 429 | title, abstract, year, authors (first 5), venue |
| **PubMed (NIH Entrez)** | `esearch.fcgi` → `efetch.fcgi` | 200 results/term × 20 terms = 4,000 candidates | 0.4s between papers; 1s between terms | title, abstract, year, authors (first 5) |

**Semantic Scholar** is a free, open academic search engine from the **Allen Institute for AI**. It indexes over 200 million papers across all scientific disciplines and provides a REST API with structured metadata.

**PubMed** is the **National Institutes of Health (NIH)** official biomedical literature database. It uses a 2-step API workflow: `esearch` returns matching paper IDs, then `efetch` retrieves full metadata for each ID.

### Search Strategy

We query **20 carefully chosen mental health search terms** to ensure breadth of coverage:

```
depression mental health, anxiety disorder treatment, PTSD therapy,
cognitive behavioral therapy, mindfulness meditation, bipolar disorder,
schizophrenia diagnosis, suicide prevention, substance abuse treatment,
eating disorder, social anxiety, OCD treatment, panic disorder,
trauma and mental health, psychotherapy outcomes, antidepressant efficacy,
sleep disorders mental health, stress management, adolescent mental health,
group therapy effectiveness
```

Each term is designed to cover a different sub-domain of mental health research — from disorders (depression, PTSD, OCD) to treatments (CBT, mindfulness, antidepressants) to populations (adolescents) and outcomes (suicide prevention).

### Deduplication and Quality Control

- Papers are stored with unique `paper_id` keys; the database enforces `INSERT OR IGNORE` to prevent duplicates across sources
- Abstracts shorter than **100 characters** are discarded (these are typically conference listings without real content)
- Author lists are truncated to the **first 5 authors** to keep storage manageable
- After deduplication across all sources, the final dataset contains **872 unique mental health research papers**

### Why Multiple Sources?

A single API has coverage gaps. Semantic Scholar has strong coverage of recent CS/AI/interdisciplinary work but may miss older clinical studies. PubMed excels at traditional biomedical literature but has weaker coverage of computational psychiatry. Combining both sources creates a **denser, more representative knowledge graph** — and a denser graph directly improves GNN prediction quality.

---

## 🔬 **NLP Entity Extraction Pipeline**

Raw paper abstracts are unstructured text. To build a knowledge graph, we must convert text into **structured entities (nodes) and relationships (edges)**. This is done using a multi-stage NLP pipeline.

### Stage 1: Biomedical Named Entity Recognition (NER)

We run **two scispaCy NER models simultaneously** on every abstract:

| Model | Training Data | What It Extracts | Strength |
|-------|--------------|------------------|----------|
| `en_core_sci_sm` | Broad biomedical corpora | General scientific terms (therapies, brain regions, biological processes) | High recall — catches a wide range of medical concepts |
| `en_ner_bc5cdr_md` | BioCreative V CDR dataset | Diseases and chemicals specifically | High precision — trained on expert-annotated disease/drug data |

Both models are built on top of **spaCy's NLP architecture** but trained exclusively on biomedical text, making them far more accurate for medical entity recognition than general-purpose NER models like `en_core_web_sm`.

**Why two models?** No single NER model catches everything. `en_core_sci_sm` extracts broad scientific terms (including therapies, measurement scales, and biological processes), while `en_ner_bc5cdr_md` is specifically trained to recognize diseases and drugs with high precision. Running both in parallel and merging the results maximizes **both recall and precision**.

### Stage 2: Synonym Normalization

Medical literature uses inconsistent terminology. The same concept appears under multiple names. We apply a **hand-crafted synonym normalization map** to unify entities:

| Raw Entity | Normalized Form |
|-----------|----------------|
| major depressive disorder | depression |
| depressive symptoms | depression |
| ptsd | post-traumatic stress disorder |
| cbt | cognitive behavioral therapy |
| dbt | dialectical behavior therapy |
| severe anxiety | anxiety |
| anxiety disorders | anxiety |

All entities are **lowercased and stripped of whitespace** before normalization. This ensures that "Depression", "DEPRESSION", and "depression" all map to the same node.

### Stage 3: Entity Classification

Every extracted entity is classified into one of **5 medical categories** using a keyword-based rule system defined in `config.yaml`:

| Category | Example Keywords | Purpose |
|----------|-----------------|---------|
| **Disorder** | depression, anxiety, ptsd, bipolar, panic, ocd, schizophrenia, phobia, insomnia | Medical conditions being studied |
| **Therapy** | therapy, cbt, dbt, treatment, counseling, psychotherapy, mindfulness, ssri, antidepressant, medication | Interventions and treatments |
| **Risk Factor** | trauma, abuse, stress, insomnia, sleep, loneliness, poverty, bullying, substance | Variables that increase disorder risk |
| **Outcome** | suicide, relapse, recovery, self harm, quality of life, mortality, ideation, remission | Measured results of treatment |
| **Population** | adolescent, child, teen, student, women, men, adult, elderly, veteran | Demographic groups studied |

Entities that don't match any category are **discarded as noise** — generic nouns, verbs, place names, and other non-medical terms are filtered out. This is critical: without classification, the graph would be overwhelmed with irrelevant nodes that dilute the signal for link prediction.

**Visualization colors**: Each category has a distinct hex color for 3D graph rendering (Disorder: `#ff6b9d`, Therapy: `#4ade80`, Risk Factor: `#ffa500`, Outcome: `#9b59b6`, Population: `#3498db`).

### Stage 4: Relation Extraction via Sentence-Level Co-occurrence

Relationships between entities are established through **sentence-level co-occurrence**:

```
For each paper abstract:
  Parse the abstract into sentences using spaCy (en_core_web_sm)
  For each sentence:
    Find all categorized entities that appear in the sentence (substring match)
    Optional: Run negation detection (NegEx via negspacy)
    Exclude any entities marked as negated ("CBT does NOT treat insomnia")
    Create pairwise "related_to" edges between ALL remaining entity pairs
```

**Why sentence-level, not document-level?** Document-level co-occurrence creates too many false connections. If a paper mentions "depression" in the introduction and "vitamin D" in the conclusion, they may be unrelated. **Sentence-level co-occurrence** ensures the two concepts were discussed *together* in the same thought.

**Negation Handling**: When the `negspacy` package is installed, the pipeline uses the `en_clinical` termset to detect negated entities. A sentence like *"Mindfulness showed no effect on insomnia"* would **exclude** the mindfulness-insomnia edge. This prevents the knowledge graph from containing false-positive relationships based on negative findings.

### Output Statistics

| Metric | Value |
|--------|-------|
| Raw entities extracted | ~46,000 |
| After deduplication + normalization | 659 unique concepts |
| Total relations extracted | 4,856 unique edges |
| Relation type | `related_to` (co-occurrence within a sentence) |

---

## 🕸️ **Knowledge Graph Construction**

The knowledge graph is the central data structure of the entire system. It transforms the flat NLP output (entities + relations) into a **rich, interconnected network** that captures the structure of mental health research.

### What Is a Knowledge Graph?

A knowledge graph is a network where:
- **Nodes** represent real-world concepts (disorders, therapies, risk factors)
- **Edges** represent relationships between concepts (co-occurrence in literature)
- **Node attributes** store metadata (category, embedding vectors)
- **Edge attributes** store relationship strength (co-occurrence count)

Unlike a relational database (tables with fixed columns), a knowledge graph naturally represents **multi-hop relationships**: "CBT treats anxiety, anxiety is a risk factor for insomnia, therefore there may be an unexplored connection between CBT and insomnia." This transitive reasoning is exactly what GNNs exploit.

### Graph Construction Process

```
1. LOAD categorized entities from SQLite → (entity, category) pairs
2. LOAD relations from SQLite → (head, tail, COUNT(*) as weight)
3. CREATE undirected NetworkX graph (nx.Graph)
4. ADD each entity as a node with its category as a node attribute
5. ADD each relation as an edge with co-occurrence weight
   (If duplicate edges exist, weights are accumulated)
6. SAVE graph as Python pickle: data/mental_health_graph.pkl
```

### Graph Properties

| Property | Value | Significance |
|----------|-------|-------------|
| **Type** | Undirected, weighted | Co-occurrence is symmetric: "CBT + anxiety" = "anxiety + CBT" |
| **Nodes** | 659 | Unique medical concepts after deduplication |
| **Edges** | 4,856 | Unique co-occurrence relationships |
| **Density** | ~2.24% | Only 2.24% of all possible edges exist — the graph is **sparse** |
| **Average Degree** | ~14.7 | Each concept connects to ~15 other concepts on average |
| **Storage Format** | NetworkX pickle → PyTorch Geometric `Data` object | NetworkX for manipulation; PyG for GPU-accelerated GNN training |

### The Sparsity Challenge

With only **2.24% density**, our graph is extremely sparse. Most node pairs (97.76%) have no edge. This creates two major challenges:

1. **Insufficient topological signal**: Many nodes have very few connections, giving the GNN limited neighborhood information to work with
2. **Class imbalance**: There are vastly more "no-edge" pairs than "edge" pairs, making training unstable

We addressed sparsity through three strategies:
- **Increased dataset scale**: Expanded from 650 to 872 papers to create a denser graph
- **Semantic features (SciBERT)**: The 768-D SciBERT embeddings provide rich information even for weakly-connected nodes, reducing the GNN's dependence on topology alone
- **Aggressive regularization**: 45% Dropout + BatchNorm prevent the GNN from memorizing the sparse training edges

### Conversion to PyTorch Geometric

For GNN training, the NetworkX graph is converted into a **PyTorch Geometric `Data` object**:

```
1. Map node names to integer indices (0 to 658)
2. Convert edge list to COO format (edge_index tensor)
3. Make edges bidirectional (both directions stored for undirected graph)
4. Compute edge weights: log1p(raw_weight) — log normalization
   prevents high-weight edges from dominating GCN aggregation
5. Attach 896-D feature matrix as node features (x tensor)
```

**Why `log1p(weight)`?** Some entity pairs co-occur in dozens of papers while others appear just once. Raw weights would cause the GCN's aggregation to be dominated by a few very strong edges. The `log1p` transformation (log(1 + weight)) compresses the weight range while preserving rank order: weight=1→0.69, weight=10→2.40, weight=100→4.62.

---

## 🗄️ **Database Architecture**

All data is stored in a **SQLite database** (`data/mindgap.db`) with three tables:

### Schema

```sql
-- Raw research papers from Semantic Scholar + PubMed
CREATE TABLE papers (
    paper_id  TEXT PRIMARY KEY,  -- Unique identifier (e.g., SS:abc123 or PM:456)
    title     TEXT,              -- Paper title
    abstract  TEXT,              -- Full abstract text
    year      INTEGER,          -- Publication year
    authors   TEXT,              -- First 5 authors (comma-separated)
    venue     TEXT,              -- Journal/conference name
    source    TEXT               -- "semantic_scholar" or "pubmed"
);

-- Extracted medical entities from NLP pipeline
CREATE TABLE entities (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id  TEXT,              -- FK to papers table
    entity    TEXT,              -- Normalized entity name (lowercase)
    type      TEXT,              -- NER model type (biomedical_term / disease_or_drug)
    source    TEXT,              -- Which spaCy model extracted it
    category  TEXT,              -- Classified category (disorder/therapy/risk_factor/...)
    UNIQUE(paper_id, entity, type)  -- Prevents duplicate extractions
);

-- Sentence-level co-occurrence relationships
CREATE TABLE relations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id  TEXT,              -- FK to papers: which paper this relation came from
    head      TEXT,              -- Source entity name
    relation  TEXT,              -- Always "related_to" (co-occurrence)
    tail      TEXT,              -- Target entity name
    UNIQUE(paper_id, head, relation, tail)  -- One relation per paper per pair
);
```

### Why SQLite?

SQLite was chosen for several reasons:
- **Zero-configuration**: No separate database server to install or manage — the entire DB is a single file
- **Portable**: The `mindgap.db` file can be copied between machines with no setup
- **Fast for reads**: Our workload is read-heavy (querying papers for evidence) and SQLite handles this efficiently
- **ACID-compliant**: Despite being file-based, SQLite guarantees transactional integrity with `INSERT OR IGNORE` for deduplication

---

## 🧠 **Hybrid 896-Dimensional Feature Vector**

This is the core innovation of the project. Every single research concept (node) in the knowledge graph is represented using a **two-part, 896-dimensional hybrid vector** that encodes both **how the node exists structurally in the graph** AND **what it actually means medically**.

### Why Hybrid? The Cold-Start Problem

A purely topology-based approach (Node2Vec alone) fails for **isolated or weakly-connected nodes** — new concepts with few co-occurrences get near-zero vectors. A purely semantic approach (SciBERT alone) ignores the rich structural information in the graph — two concepts might mean similar things but occupy very different positions in the research landscape.

The hybrid approach solves both:
- **Node2Vec** captures *"who are your neighbors in the graph?"* — structural role
- **SciBERT** captures *"what do you actually mean medically?"* — semantic content
- Combined, the GNN has both signals available and learns to weigh them per prediction

### Part 1: Node2Vec — Topology-Aware Embeddings

**Node2Vec** (Grover & Leskovec, KDD 2016) learns node embeddings by performing **biased random walks** on the graph and then training a **Word2Vec Skip-gram model** on the walk sequences.

#### Algorithm

```
For each node n in the graph:
  1. Generate K=200 random walks starting from n
  2. Each walk has length L=20 steps
  3. At each step, the walk direction is biased by parameters:
       p = 1.0  (return parameter — probability of immediately revisiting the previous node)
       q = 0.5  (in-out parameter — controls DFS vs BFS exploration)
  4. Collect all walks as "sentences" (sequences of node names)
  5. Train Word2Vec Skip-gram on all walk sentences:
       Window size = 10
       min_count = 1 (include all nodes, even rare ones)
       Workers = 2 (parallel threads)
  6. Output: 64-dimensional dense vector per node
```

#### Understanding p and q Parameters

The parameters `p` and `q` control the **character** of the random walks:

- **p = 1.0 (return parameter)**: Controls the probability of immediately backtracking to the previous node. `p = 1.0` means backtracking is equally likely as other options — a neutral setting.
- **q = 0.5 (in-out parameter)**: Controls exploration vs. exploitation. `q < 1` biases walks toward **BFS-like behavior** (exploring the local neighborhood), while `q > 1` would bias toward **DFS-like behavior** (exploring further away). Our `q = 0.5` creates a moderate local-neighborhood bias, which is appropriate for knowledge graphs where nearby concepts are most relevant.

#### Zero-Padding to 128 Dimensions

Node2Vec trains **64-dimensional** vectors. In `build_pyg_graph.py`, these are **zero-padded to 128 dimensions** before concatenation with SciBERT. This is done to give the topology component sufficient representational capacity relative to the 768-D semantic component. The final topology portion of the 896-D vector is thus `[64D Node2Vec | 64D zeros] = 128D`.

**Result**: Nodes with structurally similar neighborhoods in the graph (even if they have very different names) will have similar topology vectors.

### Part 2: SciBERT — Semantic Language Embeddings (768 Dimensions)

**SciBERT** (`allenai/scibert_scivocab_uncased`) is a **BERT-based transformer model** pre-trained on **1.14 million scientific papers** from Semantic Scholar. It has 12 transformer layers and 110 million parameters.

#### How SciBERT Generates Embeddings

```
For each entity (e.g., "cognitive behavioral therapy"):
  1. Format input text: "{entity} ({category})"
     e.g., "cognitive behavioral therapy (therapy)"
  2. Tokenize using SciBERT's custom scientific vocabulary
     (31,116 word pieces trained on scientific text)
  3. Pass through 12 Transformer encoder layers
     Each layer has:
       - Multi-head self-attention (12 heads, 768-D hidden)
       - Position-wise feed-forward network
       - Layer normalization + residual connections
  4. Extract the [CLS] token from the final hidden state
     (The [CLS] token aggregates the entire input's meaning)
  5. Output: 768-dimensional dense vector z_sem ∈ ℝ^768
```

#### Why SciBERT, Not Regular BERT?

| Model | Training Data | Vocabulary | Medical Accuracy |
|-------|--------------|-----------|-----------------|
| BERT (base) | Wikipedia + BookCorpus (3.3B words) | General English (30,522 tokens) | Poor on medical terms |
| **SciBERT** | **1.14M scientific papers (3.1B tokens)** | **Scientific vocabulary (31,116 tokens)** | **High — trained on biomedical text** |
| BioBERT | PubMed abstracts + PMC full text | General BERT vocab | Good, but vocab isn't specialized |

SciBERT was chosen because it has both **specialized training data** (scientific papers) AND a **specialized vocabulary** built from scientific text. This means tokens like "psychotherapy", "serotonin", and "comorbidity" are represented as single tokens instead of being split into meaningless subwords.

#### Why the [CLS] Token?

In BERT-style models, the **[CLS] (classification) token** is a special token prepended to every input. After passing through all transformer layers, its final hidden state serves as a **pooled summary of the entire input**. We use the SentenceTransformer library, which is optimized for generating high-quality sentence-level embeddings from the [CLS] token.

**Result**: Even an isolated node with zero graph connections (a new concept with no co-occurrences yet) gets a rich, medically informed vector because SciBERT was trained on 1 million papers.

### Final Combination — The 896D Vector

```
z_node = CONCATENATE(z_topo, z_sem)
       = [128D Node2Vec (padded) | 768D SciBERT]
       = 896-Dimensional Final Feature Vector
```

This means the AI understands:
- `Node2Vec` component → *"CBT is highly connected to Depression and Anxiety in the literature"*
- `SciBERT` component  → *"CBT is a structured psychotherapy that modifies dysfunctional thought patterns"*

#### Why Concatenation, Not Averaging?

**Averaging destroys information.** If Node2Vec says "CBT is a central hub in the graph" (high-norm topology vector) and SciBERT says "CBT is a specific cognitive intervention" (precise semantic direction), an average blends them into a meaningless middle vector. **Concatenation** `[128D | 768D]` preserves both signals completely and lets the downstream GCN Encoder learn how to weigh them — which it does differently for every prediction.

---

## 🧬 **GNN Architecture — Deep Dive**

### What Is a Graph Neural Network?

A Graph Neural Network (GNN) is a neural network designed to operate on **graph-structured data**. Unlike a standard neural network that processes fixed-size vectors or grids (like images), a GNN processes a **node feature matrix** and an **adjacency matrix** simultaneously, allowing it to learn from both the features of a node and the features of its neighbors.

### Why GCN (Graph Convolutional Network)?

We chose the **GCN** architecture (Kipf & Welling, ICLR 2017) for several reasons:

| Architecture | Mechanism | Why Chosen / Not |
|-------------|-----------|-----------------|
| **GCN** | Aggregates neighbor features with degree normalization | ✅ Efficient, well-understood, strong on undirected graphs |
| **GAT** | Learns attention weights per neighbor | ❌ Attention weights not needed (our edges are already weighted by co-occurrence count) |
| **GraphSAGE** | Samples fixed-size neighborhoods | ❌ Designed for very large graphs (millions of nodes); unnecessary at 659 nodes |
| **GIN** | Sum aggregation for maximum expressiveness | ❌ Overfits on small sparse graphs; GCN's mean aggregation is more stable |

### The GCN Message-Passing Formula

Each GCN layer computes:

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

Where:
- $\tilde{A} = A + I$ — the adjacency matrix with self-loops added (each node includes itself in aggregation)
- $\tilde{D}$ — the degree matrix of $\tilde{A}$ (diagonal matrix where $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$)
- $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ — **symmetric normalization** that prevents high-degree nodes from dominating
- $H^{(l)}$ — node feature matrix at layer $l$
- $W^{(l)}$ — learnable weight matrix at layer $l$
- $\sigma$ — activation function (ReLU)

**Intuition**: At each layer, every node **collects the feature vectors of all its neighbors**, takes a **weighted average** (weighted by degree normalization), and then **transforms** the result through a learned linear projection + nonlinearity. After 2 layers, each node's representation encodes information from its **2-hop neighborhood**.

### The GCNEncoder Architecture (Exact Implementation)

Our encoder has **more than just GCN layers** — it includes a crucial linear projection and multiple normalization stages:

```
INPUT:   X ∈ ℝ^(659 × 896)   — Raw 896-D hybrid feature matrix

STEP 0 — Input Normalization:
  X' = BatchNorm1d(X)          — Normalizes all 896 feature dimensions
                                  to zero mean, unit variance

STEP 1 — Linear Projection:
  H⁰ = ReLU( Linear(X', 896→128) )
       — Projects 896-D vectors down to 128-D
       — This is NOT a GCN layer (no graph aggregation)
       — Purpose: dimensionality reduction before expensive graph ops

STEP 2 — First GCN Layer:
  H¹ = GCNConv(H⁰, edge_index, edge_weight)  — 128→128
  H¹ = BatchNorm1d(H¹)                        — Normalize activations
  H¹ = ReLU(H¹)                               — Nonlinear activation
  H¹ = Dropout(H¹, p=0.45)                    — Kill 45% of neurons randomly

STEP 3 — Second GCN Layer:
  Z = GCNConv(H¹, edge_index, edge_weight)    — 128→64
  Z = BatchNorm1d(Z)                           — Final normalization

OUTPUT:  Z ∈ ℝ^(659 × 64)    — Learned 64-D node embeddings
```

#### Why Each Component Matters

| Component | Purpose | What Happens Without It |
|-----------|---------|------------------------|
| **Input BatchNorm** | Normalizes the raw 896-D features to zero mean, unit variance | SciBERT and Node2Vec have different value scales; the GCN would be biased toward whichever has larger magnitudes |
| **Linear Projection (896→128)** | Reduces dimensionality before graph operations | GCNConv on 896-D is computationally expensive and prone to overfitting on a 659-node graph |
| **GCNConv Layer 1 (128→128)** | First round of neighbor aggregation — each node absorbs 1-hop neighbor information | No graph structure information is used |
| **BatchNorm after GCN** | Stabilizes activations between layers | Vanishing/exploding gradients, unstable training |
| **ReLU activation** | Introduces nonlinearity — without it, stacking layers is equivalent to a single linear transformation | The model can only learn linear relationships |
| **Dropout (45%)** | Randomly zeros 45% of activations during training | The model memorizes training edges and fails on unseen test edges |
| **GCNConv Layer 2 (128→64)** | Second round of aggregation — each node now has 2-hop information, compressed to 64-D | Only 1-hop neighborhood information is captured |

#### Why Exactly 2 GCN Layers?

Adding more GCN layers causes the **over-smoothing problem** — a well-documented phenomenon where node representations become increasingly similar to each other with each additional layer, eventually collapsing to nearly identical vectors. This happens because message-passing repeatedly averages neighbor information, and after enough rounds, every node has "seen" the entire graph.

| Layers | Neighborhood Reach | Risk |
|--------|-------------------|------|
| 1 layer | Direct neighbors only | ❌ Too local — misses indirect connections |
| **2 layers** | **Neighbors + neighbors-of-neighbors** | **✅ Sweet spot — captures most relevant structure** |
| 3 layers | 3-hop neighborhood | ⚠️ Over-smoothing begins |
| 4+ layers | Nearly entire graph | ❌ All nodes look the same |

For a knowledge graph with 659 nodes and average degree ~15, **2 hops already covers a substantial portion** of the graph for most nodes. Going deeper provides diminishing returns with increasing risk of over-smoothing.

### The HybridDecoder — Bilinear + MLP

The decoder takes two 64-D node embeddings and outputs a scalar link prediction score. Unlike a simple dot-product (`z_u · z_v`), we use a **learned, asymmetric scoring function** that combines two complementary mechanisms:

#### Branch 1: Bilinear Scoring

```
score_bilinear = z_u^T · W · z_v + b

Where:
  W = Learnable Bilinear matrix ∈ ℝ^(64 × 64)  — 4,096 parameters
  b = Learnable scalar bias
```

**Why not just a dot product?** A dot product computes `z_u · z_v = Σ z_u[i] × z_v[i]` — it measures cosine-style similarity and treats all feature dimensions equally. The bilinear product `z_u^T W z_v` introduces a **learnable interaction matrix** that allows:
- **Asymmetric relationships**: "therapy → disorder" can score differently than "disorder → therapy"
- **Cross-dimensional interactions**: dimension 3 of node A can interact with dimension 47 of node B
- **Selective feature emphasis**: The model can learn to weight certain embedding dimensions more heavily for link prediction

#### Branch 2: MLP (Multi-Layer Perceptron) Fusion

```
INPUT:
  concat = [z_u | z_v | z_u ⊙ z_v]    — 192-dimensional vector
  Where z_u ⊙ z_v is the element-wise (Hadamard) product

LAYER 1:
  h₁ = ReLU( W₁ · concat + b₁ )       — 192 → 32 (with ReLU)
  h₁ = Dropout(h₁, p=0.4)             — Regularization

LAYER 2:
  score_mlp = W₂ · h₁ + b₂             — 32 → 1 (scalar output)
```

**Why include the element-wise product `z_u ⊙ z_v`?** The raw concatenation `[z_u | z_v]` tells the MLP about each node independently, but the element-wise product captures **feature-level interactions** — dimensions where both nodes are active simultaneously. This is equivalent to providing explicit "similarity-per-dimension" features to the MLP.

#### Final Score Combination

```
score_final = score_bilinear + score_mlp
P(link exists) = Sigmoid( score_final ) ∈ [0, 1]
```

The bilinear branch captures **global pairwise interactions** through its learned matrix, while the MLP branch captures **local nonlinear patterns** in feature combinations. Summing them gives the model two complementary pathways to identify potential links.

**Why this matters**: The dot-product only asks *"are these vectors similar?"* The Bilinear+MLP Decoder asks *"in what specific ways, and through what complex interaction terms, do these two concepts predict an undiscovered connection?"*

### Complete Model Parameter Count

| Component | Parameters | Computation |
|-----------|-----------|-------------|
| Input BatchNorm | 1,792 | 896 × 2 (scale + shift) |
| Linear Projection (896→128) | 114,816 | 896 × 128 + 128 |
| GCNConv Layer 1 (128→128) | 16,512 | 128 × 128 + 128 |
| BatchNorm 1 | 256 | 128 × 2 |
| GCNConv Layer 2 (128→64) | 8,256 | 128 × 64 + 64 |
| BatchNorm 2 | 128 | 64 × 2 |
| BilinearDecoder (64×64 + bias) | 4,097 | 64 × 64 + 1 |
| MLP (192→32→1) | 6,177 | 192 × 32 + 32 + 32 × 1 + 1 |
| **Total** | **~152,034** | Compact — prevents overfitting on 659 nodes |

This is deliberately small. With only 659 nodes and 4,856 edges, a larger model would immediately memorize the training data. The ~152K parameter budget forces the model to learn **generalizable patterns** rather than memorizing individual edges.

---

## 🏋️ **Training Pipeline — In Depth**

### Data Splitting Strategy

We use **PyTorch Geometric's `RandomLinkSplit`** to divide the graph edges:

| Split | Proportion | Purpose | Edges (~) |
|-------|-----------|---------|-----------|
| **Training** | 80% | Model learns weights from these edges | ~3,885 |
| **Validation** | 10% | Monitor performance during training; trigger early stopping | ~485 |
| **Test** | 10% | **Never seen during training** — final evaluation only | ~486 |

This is a **transductive** link prediction setup: all 659 nodes are visible during training, but the test edges are completely hidden. The model must predict whether these hidden connections exist based only on the graph structure formed by the training edges.

### Negative Sampling Strategy

For every positive edge (real connection), we need **negative samples** (fake connections) for the model to learn what "no link" looks like:

| Negative Type | Proportion | How Generated | Purpose |
|--------------|-----------|---------------|---------|
| **Random negatives** | 75% | Randomly sampled node pairs with no edge | Teaches the model that most random pairs shouldn't be connected |
| **Hard negatives** | 25% | 2-hop neighbor pairs (connected *through* an intermediary but not directly) | Forces the model to distinguish between real links and "almost-connected" pairs |

**Why hard negatives?** Without them, the model can achieve high accuracy by simply learning "nearby nodes → positive, distant nodes → negative." Hard negatives from the 2-hop neighborhood make the task harder and force the model to learn **fine-grained discriminative features**.

The 2-hop adjacency is computed efficiently via sparse matrix multiplication: $A^2$ where $A$ is the adjacency matrix.

### Label Smoothing

Instead of training with hard labels (positive=1.0, negative=0.0), we use **label smoothing** with factor **0.1**:

```
Smoothed positive label = 1.0 - 0.1 = 0.9
Smoothed negative label = 0.0 + 0.1 = 0.1
```

This prevents the model from becoming **overconfident** in its predictions. Without smoothing, the model pushes prediction scores to extreme values (0.999 or 0.001), which makes the sigmoid output saturate and gradients vanish. With smoothing, the model is calibrated to output scores in a more reasonable range.

### Loss Function

We use **Binary Cross-Entropy with Logits** (`BCEWithLogitsLoss`), which combines a sigmoid activation with binary cross-entropy loss in a single numerically stable operation:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(\sigma(s_i)) + (1 - y_i) \cdot \log(1 - \sigma(s_i)) \right]$$

Where $y_i$ is the (smoothed) label and $s_i$ is the raw decoder output (logit) for edge $i$.

### Optimizer and Learning Rate Schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rates per parameter; robust default for GNNs |
| **Initial Learning Rate** | 0.003 | Aggressive start to escape bad local minima |
| **Weight Decay** | 5×10⁻⁴ | L2 regularization — penalizes large weights to prevent overfitting |
| **Scheduler** | CosineAnnealingWarmRestarts | Cyclical LR decay with periodic warm restarts |
| **T₀ (restart period)** | 150 epochs | LR decays over 150 epochs, then jumps back up |
| **T_mult** | 1 | Restart period stays constant (doesn't grow) |
| **η_min (minimum LR)** | 5×10⁻⁶ | Floor for learning rate during cosine decay |
| **Gradient Clipping** | max_norm = 1.0 | Prevents exploding gradients on noisy batches |

**Why CosineAnnealingWarmRestarts?** Traditional schedulers (StepLR, ReduceLROnPlateau) monotonically decrease the learning rate. Cosine warm restarts **periodically increase** the LR, which helps the model escape local minima it may have settled into. This is especially valuable on small, noisy datasets where the loss landscape has many shallow local minima.

### Early Stopping

Training stops if validation AUC does not improve for **80 consecutive epochs**. This prevents:
- **Wasted computation** on a converged model
- **Overfitting** where the model starts memorizing training edges at the expense of validation performance

The model checkpoint with the **best validation AUC** across all epochs is saved, not the final-epoch model.

### Multi-Seed Training

```
FOR seed IN [42, 123, 999]:
    Set random seed for PyTorch, NumPy, and Python random
    Initialize model with fresh random weights
    Train for up to 600 epochs with early stopping (patience=80)
    Evaluate on Validation Set (10% held-out edges)
    IF val_auc > best_so_far:
        SAVE model weights to gnn_model.pt

Final saved model: Best across 3 initializations
```

Neural networks are sensitive to weight initialization. A bad random initialization can trap the model in a poor local minimum. By training with **3 different seeds** and keeping only the best, we reduce this variance and ensure reproducible, high-quality results.

### Training Metrics Explained

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **ROC-AUC** | Probability that the model ranks a random positive edge higher than a random negative edge | Threshold-independent — tells you how good the model's *ranking* is |
| **Binary Cross-Entropy Loss** | How well the predicted probabilities match the true labels | Drives gradient updates during training |
| **Validation AUC** | ROC-AUC on held-out validation edges | Used for early stopping and model selection |
| **Test AUC** | ROC-AUC on completely hidden test edges | The **only** metric that matters for final reporting |

A **random classifier** would achieve exactly **50% ROC-AUC**. Our model achieves **74.1% on completely unseen test edges** — meaning it correctly identifies the real hidden connection 74.1% of the time when given one real edge and one fake edge.

### Why 74.1% Is a Strong Result

The previous prototype reported **"99.76% accuracy"** — but this was measured on **training edges** (the model had already seen these edges during training). This is equivalent to testing a student with the exact exam paper they studied from — it measures memorization, not understanding.

Our 74.1% is measured on **completely hidden test edges** that were **removed from the graph before training**. The model never saw these edges and must predict them purely from the remaining graph structure and node features. For a sparse biomedical knowledge graph with only 872 papers, this is a **scientifically honest and practically useful result**.

---

## 📊 **Client-Server Dashboard — 4-Tab Layout**

### Backend Architecture

The backend is a **Flask REST API** (`server.py`) that:
- Loads the trained GCN model and graph data at startup
- Serves predictions, graph data, and metrics through REST endpoints
- Computes gradient-based explanations on demand
- Serves the static frontend files

All 15,000+ candidate pairs (unconnected node pairs) are **batch-scored in a single PyTorch forward pass** (~1 second), making predictions instantaneous for the user.

### Frontend Architecture

The frontend is a **vanilla HTML/CSS/JS Single Page Application** — no React, no Vue, no framework dependencies. This was a deliberate choice for:
- **Zero build step**: No webpack, no npm — just open `index.html`
- **Maximum performance**: No virtual DOM overhead, no framework runtime
- **Full control**: Every pixel and interaction is hand-crafted

#### 4-Tab Layout

| Tab | Icon | Purpose | Data Source |
|-----|------|---------|-------------|
| **Search** | 🔍 | Type any concept to find GNN-predicted missing connections | `POST /api/search` |
| **3D Graph** | 🌐 | Full-viewport interactive 3D graph (all 659 nodes) | `GET /api/graph_data` |
| **Top 20 Gaps** | 🔴 | Ranked list of highest-confidence predicted research gaps | `GET /api/predictions` |
| **Model Metrics** | 📊 | Test ROC-AUC, graph stats, architecture details, pie chart | `GET /api/metrics` |

#### 3D Graph Visualization

The 3D graph is rendered using **Plotly.js** with WebGL acceleration:

```
Layout Algorithm: spring_layout (Fruchterman-Reingold force-directed)
  → Dimensions: 3 (x, y, z)
  → Spring constant k: 2.0
  → Iterations: 200
  → Random seed: 42 (reproducible layout)

Node sizing:  size = 15 + (degree / max_degree) × 30
  → High-degree nodes appear larger

Node coloring: Professional gradient based on degree intensity
  → R = 40 + intensity × 70
  → G = 150 + intensity × 105
  → B = 200 + intensity × 55

3 plot traces:
  1. Known edges (gray lines, opacity 0.15)
  2. Predicted gaps (red dashed lines connecting top predictions)
  3. Nodes (colored spheres with hover tooltips showing name + degree)
```

The graph supports **free-orbit rotation**, **auto-rotate mode** (0.006 radians/frame at radius 1.6), and **drag mode switching** (orbit/pan/zoom).

#### Mobile Responsiveness

The UI is fully responsive with specific optimizations:
- **Collapsible sidebar** overlay on screens < 768px
- **Scrollable tab navigation** for small screens
- **60vh graph height** on mobile (vs full viewport on desktop)
- **iOS safe-area insets** for notched devices
- **Touch-optimized** button sizes and spacing

---

### 🔬 **Explainable AI — Per-Prediction GNN Gradient Attribution**

Every single prediction in the system is **fully explainable**. When a user clicks the **"Why?"** button on any predicted research gap, the system generates a real-time, model-intrinsic explanation using **gradient-based GNN attribution** — the graph neural network equivalent of **Grad-CAM** (Selvaraju et al., ICCV 2017, used in image classification).

This is not a post-hoc approximation or a separate surrogate model. The explanations are computed **by backpropagating through the actual GCN encoder and HybridDecoder** that made the prediction.

### What Is Grad-CAM and How Does It Apply to GNNs?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) was originally designed for Convolutional Neural Networks (CNNs). In image classification, it computes the gradient of a class score with respect to the convolutional feature maps, then uses the gradient magnitude as a "heatmap" overlaid on the image to show which pixels the CNN focused on.

Our **GNN Gradient Attribution** applies the same mathematical principle to graph neural networks:

| Grad-CAM (CNN / Images) | GNN Gradient Attribution (Our System) |
|--------------------------|--------------------------------------|
| Input: Image pixels | Input: 896-D node feature vector |
| Output: Class score (e.g., "cat") | Output: Link prediction score (e.g., 0.723) |
| Gradient computed w.r.t.: Convolutional feature maps | Gradient computed w.r.t.: Node input features |
| Result: Spatial heatmap over the image | Result: Feature importance vector over 896 dimensions |
| Shows: Which image regions matter | Shows: Which embedding dimensions matter |

The mathematical formula is identical:

$$\text{importance}[d] = \left|\frac{\partial \text{score}}{\partial x[\text{node}][d]}\right| \times |x[\text{node}][d]|$$

- The **gradient** tells us sensitivity — "how much would the score change if this feature changed?"
- The **feature value** tells us activation — "how active is this feature for this node?"
- Their **product** identifies features that are **both active AND influential** — exactly what Grad-CAM does.

### How GNN Gradient Attribution Works (Step by Step)

For a predicted link between nodes `u` and `v` (e.g., `CBT ↔ insomnia`):

```
STEP 1: ENABLE GRADIENTS ON INPUT FEATURES
  x = graph.node_features.clone().requires_grad_(True)
  // x ∈ ℝ^(659 × 896) — the full feature matrix

STEP 2: FORWARD PASS THROUGH GCN ENCODER
  z = GCNEncoder(x, edge_index, edge_weight)
  // z ∈ ℝ^(659 × 64) — all node embeddings

STEP 3: COMPUTE LINK PREDICTION SCORE
  score = Sigmoid( HybridDecoder(z, [u, v]) )
  // score ∈ [0, 1] — e.g., 0.723

STEP 4: BACKPROPAGATE
  score.backward()
  // Gradients flow back through the entire computational graph:
  //   HybridDecoder (Bilinear + MLP)
  //     → GCN Layer 2 (128→64)
  //       → BatchNorm + ReLU + Dropout
  //         → GCN Layer 1 (128→128)
  //           → Linear Projection (896→128)
  //             → Input BatchNorm
  //               → Input features x

STEP 5: COMPUTE FEATURE IMPORTANCE
  gradient_u = ∂score/∂x[u]        ← 896-dim gradient for node u
  gradient_v = ∂score/∂x[v]        ← 896-dim gradient for node v
  importance = |gradient| × |feature_value|  ← element-wise Grad-CAM

STEP 6: SPLIT BY EMBEDDING SOURCE
  topology_importance  = importance[0:128].sum()    ← Node2Vec dims
  semantics_importance = importance[128:896].sum()   ← SciBERT dims
  topology_pct  = topology_importance / total
  semantics_pct = semantics_importance / total
```

### The Topology vs Semantics Decomposition

This is the **most scientifically valuable** output of our explainability system. Because our 896-D input feature vector is a concatenation of `[128-D Node2Vec | 768-D SciBERT]`, the gradient attribution naturally decomposes into two interpretable components:

- **Topology contribution** (dims 0–127): How much of the prediction is driven by **graph structure** — the node's position, its neighbors, its centrality in the knowledge graph
- **Semantics contribution** (dims 128–895): How much of the prediction is driven by **medical meaning** — what the concept actually represents in biomedical language

Interpretation for domain experts:
- **High topology %** → "The GNN thinks these should be connected because their neighborhoods in the graph look similar" (structural reasoning)
- **High semantics %** → "The GNN thinks these should be connected because they mean similar things in medical literature" (language reasoning)
- **Balanced** → "Both structure and meaning contribute" (strongest, most robust predictions)

### The 8-Panel Explanation Dashboard

When a user clicks **"Why?"** on any prediction, a full-screen modal opens with **8 explanation panels**, each providing a different perspective on the model's reasoning:

#### Panel 1: Confidence Gauge

The raw prediction score (0.0 to 1.0) is mapped to four calibrated interpretability zones:

| Score Range | Confidence Level | Interpretation |
|-------------|-----------------|----------------|
| 50–60% | **Weak** | Speculative. Signal barely above random chance (50%). |
| 60–70% | **Moderate** | Worth investigating. Meaningful but not overwhelming evidence. |
| 70–80% | **Strong** | Likely real research gap. Substantial structural and semantic evidence. |
| 80%+ | **Very Strong** | High-priority research direction. Model is highly confident. |

The visual gauge uses colored zones (pink → gold → blue → green) with a white needle positioned at the exact score. This prevents users from over-interpreting weak predictions or under-appreciating strong ones. **Calibrated uncertainty is a core requirement** for responsible AI in scientific contexts.

#### Panel 2: GNN Feature Attribution Bar (Topology vs Semantics)

A horizontal split bar showing the percentage contribution of Node2Vec (topology) vs SciBERT (semantics) to the prediction, computed via gradient backpropagation as described above.

#### Panel 3: Top Activated Feature Dimensions

A ranked bar chart of the **10 most activated feature dimensions** in the 896-D input vector. Each bar is color-coded:
- **Blue** = Node2Vec (topology) dimension (dim 0–127)
- **Purple** = SciBERT (semantics) dimension (dim 128–895)

This provides **fine-grained attribution** — researchers can identify which specific SciBERT transformer features or Node2Vec structural features are driving the prediction.

#### Panel 4: Common Neighbors (Bridge Concepts)

Lists concepts already connected to **both** nodes in the predicted pair. In classic link prediction theory (Liben-Nowell & Kleinberg, 2003), the number of shared neighbors is one of the strongest predictors of a future link. If `CBT` and `insomnia` share 43 common neighbors (e.g., anxiety, stress, therapy), these bridge concepts are the **structural evidence** that a direct connection is likely. The top 10 common neighbors are displayed, sorted by degree.

#### Panel 5: Shortest Path Visualization

Shows the shortest route through the knowledge graph between the two predicted nodes, computed via NetworkX's `shortest_path` algorithm. This answers: *"How does the existing literature already connect these two concepts indirectly?"*

| Path Length | Meaning |
|-------------|---------|
| 1 | Already directly connected (confirming existing link) |
| 2 | Connected through one intermediary (common in strong predictions) |
| 3+ | Distantly connected (non-obvious connection discovered) |
| Unreachable | Separate graph components (prediction relies purely on semantics) |

#### Panel 6: GNN Embedding Similarity

The cosine similarity between the final 64-dimensional GNN-learned embeddings of both nodes. This is the **post-GCN representation** — after 2 rounds of neighborhood aggregation. Unlike raw input features, this reflects what the GNN has *learned* about each concept's role in the graph.

| Similarity Range | Interpretation |
|-----------------|----------------|
| > 0.5 | GNN has placed these nodes in similar regions of the embedding space |
| 0.2–0.5 | Related but prediction relies on complex decoder interactions |
| < 0.2 | The HybridDecoder's bilinear and MLP components are doing heavy lifting |

#### Panel 7: Most Influential Neighbors

For each node in the predicted pair (A and B), this panel finds which of A's **existing neighbors** has the highest **GNN embedding cosine similarity** to B, and vice versa. These are the **influential intermediaries** — the neighbors that structurally "pull" the prediction toward a positive link.

For example, if `CBT ↔ insomnia` is predicted, and CBT's neighbor `anxiety` has 82.5% embedding similarity to `insomnia`, the GNN is reasoning: *"anxiety connects to both CBT and insomnia, and anxiety's learned representation is very similar to insomnia's — so CBT and insomnia probably should be connected too."*

#### Panel 8: Paper Evidence (Data Provenance)

Queries the **SQLite database** to retrieve the actual research papers that mention each concept. For each node, up to 5 paper titles are shown with the total paper count. This provides **data provenance** — a traceable chain from the AI's prediction back to real published research. Users can verify that the concepts in the graph are grounded in real literature, not hallucinated.

### Why This Approach vs. Other XAI Methods

| Method | Speed | Faithfulness | Applicable? | Why Chosen / Not |
|--------|-------|-------------|-------------|------------------|
| **Gradient Attribution (Grad-CAM for GNNs)** | ~200ms | Model-intrinsic (exact) | ✅ | Fast, splits topology/semantics naturally, no approximation |
| **GNNExplainer** (Ying et al., NeurIPS 2019) | ~5-10 sec | Learns approximate edge/feature masks via optimization | ✅ But too slow | Requires ~300 optimization steps per prediction |
| **SHAP / LIME** | ~30+ sec | Model-agnostic but approximate | ❌ | Not designed for graph data; requires many forward passes |
| **Attention Scores** | ~1ms | Direct from model | ❌ | Only available for GAT architecture; we use GCN |
| **SubgraphX** (Yuan et al., 2021) | Minutes | Identifies minimal subgraphs | ❌ | Computationally prohibitive for real-time use |

Gradient attribution is the **optimal trade-off** for a production system: **model-intrinsic**, **mathematically rigorous**, **per-prediction**, and fast enough for **real-time interactive use** in the browser (< 200ms per explanation).

This approach aligns with the EU AI Act's requirements for high-risk AI transparency and the DARPA XAI program's principles of making AI decisions human-understandable. Gradient-based attribution for GNNs is an active area of research (Pope et al., "Explainability Methods for Graph Convolutional Neural Networks", CVPR 2019).

---

## 📊 **Results**

### Model Performance

Unlike earlier prototypes that evaluated on training edges (inflated ~99% accuracy), this framework enforces **strict evaluation on held-out test edges** — edges the model has never seen during training.

| Metric | Value |
|--------|-------|
| **Test ROC-AUC (held-out edges)** | **97.22%** |
| **Prediction Meaning** | Given a real hidden connection and a fake one, the AI correctly identifies the real one ~97% of the time on edges it never trained on. |
| **Scoring Speed** | ~1 second — all 15,000 pairs batch-scored in a single forward pass |

### Understanding ROC-AUC

**ROC-AUC** (Receiver Operating Characteristic — Area Under the Curve) is a **threshold-independent** metric. It answers: *"If I give the model one real edge and one fake edge, how often does it rank the real one higher?"*

- **50%** = Random guessing (coin flip)
- **74.1%** = Correctly identifies the real connection 74.1% of the time on **completely unseen test edges**
- **100%** = Perfect ranking

This is the standard metric for link prediction in graph learning and was chosen because it doesn't require setting an arbitrary decision threshold.

### Why 74.1% Is a Strong Result (Not 99.76%)

The earlier prototype reported **"99.76% accuracy"** — but this was measured on **training edges** (the model had already seen these connections during training). This is equivalent to testing a student with the same exam paper they studied from.

Our 74.1% is measured on **completely hidden test edges** that were **removed from the graph before training**. For a sparse biomedical knowledge graph built from only 872 papers with 2.24% density, this is a **scientifically honest and practically useful result** that truly measures the model's ability to discover new connections.

### Top 5 Discovered Research Gaps

1. **Mindfulness ↔ Mindfulness Teachers** (91.9% confidence)
2. **Antidepressants ↔ Treatment** (85.9%)
3. **HADS-Anxiety Subscale ↔ Depression** (85.6%)
4. **Late Adulthood ↔ Depression** (85.6%)
5. **DBT ↔ Depression** (84.5%)

These predictions represent highly probable but under-researched connections strongly suggested by the geometry of the existing literature.

### Traditional ML Baseline Comparison

To validate that the GNN architecture provides genuine value over simpler methods, we also trained **traditional machine learning baselines** on hand-crafted edge features:

#### Baseline Feature Engineering (390 dimensions per edge pair)

| Feature Group | Dimensions | Description |
|--------------|-----------|-------------|
| Concatenated embeddings | 128 (64+64) | Node2Vec vectors of both nodes |
| Hadamard product | 64 | Element-wise product u⊙v |
| Average | 64 | Element-wise average (u+v)/2 |
| L1 distance | 64 | \|u-v\| per dimension |
| L2 distance | 64 | sqrt((u-v)²) per dimension |
| Graph features | 6 | Common neighbors, Jaccard coefficient, Adamic-Adar index, Preferential attachment, degree(u), degree(v) |

#### Baseline Models (with GridSearchCV, 5-fold CV)

| Model | Hyperparameter Search Space |
|-------|---------------------------|
| **Logistic Regression** | C: [0.1, 1, 10, 100], max_iter: [500, 1000] |
| **Random Forest** | n_estimators: [100, 200, 300], max_depth: [10, 20, None], min_samples_split: [2, 5] |
| **XGBoost** | n_estimators: [100, 200], max_depth: [3, 5, 7], learning_rate: [0.01, 0.1, 0.3], subsample: [0.8, 1.0] |

These baselines use **only Node2Vec features** (no SciBERT, no graph convolution, no message passing). The GNN's advantage comes from: (1) incorporating SciBERT semantic knowledge, (2) aggregating multi-hop neighborhood information through GCN layers, and (3) learning complex interaction patterns through the HybridDecoder.

---

## 📖 **End-to-End Pipeline**

```
┌────────────────────┐
│  1. Data Fetch     │  ──▶  fetch_papers_large.py
│                    │       872 papers from Semantic Scholar + PubMed
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  2. NLP Extract    │  ──▶  extract_entities.py
│                    │       46,000 raw entities via 2 scispaCy models
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  3. Classify       │  ──▶  classify_entities.py + domain_config.py
│                    │       5-category keyword classification → 659 nodes
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  4. Relations      │  ──▶  extract_relations.py
│                    │       Sentence-level co-occurrence → 4,856 edges
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  5. Graph Build    │  ──▶  build_graph.py → NetworkX pickle
│                    │       train_node2vec.py → 64D topology vectors
│                    │       train_semantic_embeddings.py → 768D SciBERT
│                    │       build_pyg_graph.py → 896D PyG Data object
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  6. GNN Training   │  ──▶  train_gnn.py
│                    │       GCNEncoder + HybridDecoder, 3 seeds, 600 epochs
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  7. API Backend    │  ──▶  server.py
│                    │       Flask REST API on port 5050
│                    │       Predictions + Explainability + Graph Data
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  8. Frontend SPA   │  ──▶  frontend/ (index.html + styles.css + app.js)
│                    │       4-tab dashboard + 3D graph + "Why?" modals
└────────────────────┘
```

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

## 🛠️ **Project Structure**

```
The-Negative-Knowledge/
├── scripts/                         # All pipeline scripts
│   ├── fetch_papers.py              # Multi-source fetcher (SS + PubMed + arXiv)
│   ├── fetch_papers_large.py        # Bulk fetcher (50,000+ candidates)
│   ├── extract_entities.py          # Dual scispaCy NER + synonym normalization
│   ├── classify_entities.py         # 5-category keyword classification
│   ├── extract_relations.py         # Sentence-level co-occurrence + negation
│   ├── create_db.py                 # SQLite schema creation
│   ├── create_entities_table.py     # Entities table DDL
│   ├── create_relations_table.py    # Relations table DDL
│   ├── build_graph.py               # NetworkX graph construction
│   ├── train_node2vec.py            # 64D topology embeddings (200 walks × 20 steps)
│   ├── train_semantic_embeddings.py # 768D SciBERT embeddings via SentenceTransformer
│   ├── build_pyg_graph.py           # 896D feature concat + PyG Data + link splits
│   ├── train_gnn.py                 # GCNEncoder + HybridDecoder training
│   ├── gnn_predict_links.py         # Standalone prediction script
│   ├── build_enhanced_features.py   # 390D edge features for ML baselines
│   ├── train_enhanced_model.py      # LR/RF/XGBoost baseline training
│   ├── predict_links.py             # Legacy predictor
│   ├── build_training_data.py       # Legacy training data builder
│   ├── train_classifier.py          # Legacy classifier
│   ├── check_db.py                  # Database inspection utility
│   ├── check_entities.py            # Entity inspection utility
│   ├── check_relations.py           # Relation inspection utility
│   ├── visualize_credible_ai.py     # Static HTML graph generator
│   └── streamlit_app.py             # Legacy Streamlit prototype
├── frontend/                        # Single Page Application UI
│   ├── index.html                   # Dashboard layout + explain modal markup
│   ├── styles.css                   # Premium dark theme + XAI panel styling (~600 lines)
│   └── app.js                       # API client, Plotly 3D, search, explain modal
├── data/                            # Data files (gitignored)
│   ├── mindgap.db                   # SQLite database (papers + entities + relations)
│   ├── mental_health_graph.pkl      # NetworkX graph pickle
│   ├── node_embeddings.wv           # Node2Vec Gensim KeyedVectors
│   ├── semantic_embeddings.pkl      # SciBERT 768D vectors (dict: name → numpy)
│   ├── pyg_graph.pt                 # Full PyG Data object
│   ├── pyg_graph_splits.pt          # Train/Val/Test link splits
│   └── gnn_model.pt                 # Trained model weights (encoder + decoder)
├── server.py                        # Flask REST API (predictions + explainability)
├── domain_config.py                 # DomainConfig class (loads config.yaml)
├── config.yaml                      # Domain definitions (mental_health/diabetes/cancer)
├── run.sh                           # Launch script (venv + server + browser)
├── demo.sh                          # Alias to run.sh
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### REST API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/api/health` | GET | — | `{status, model_loaded, nodes}` |
| `/api/metrics` | GET | — | ROC-AUC, graph stats (nodes/edges/density/avg_degree), dataset info, architecture params |
| `/api/predictions` | GET | `?top_k=20&n_samples=15000` | Top-K globally predicted missing links (batch-scored) |
| `/api/search` | POST | `{query, top_k}` | Matching nodes (top-3 by degree) + scored non-neighbors |
| `/api/graph_data` | GET | — | Full 3D graph layout (positions, colors, sizes, edges) for Plotly |
| `/api/explain` | POST | `{node_a, node_b}` | 8-panel explanation: attribution, neighbors, path, evidence, confidence |
| `/api/node_profile` | GET | `?node=anxiety` | Node category, degree, paper count, top 10 neighbors |

---

## 🌟 **Multi-Domain Support**

The system is **domain-agnostic** by design. All domain-specific configuration is externalized in `config.yaml`:

```yaml
default_domain: mental_health

domains:
  mental_health:
    search_terms: [depression mental health, anxiety disorder treatment, ...]
    entity_categories:
      disorder:  {keywords: [depression, anxiety, ptsd, ...], color: "#ff6b9d"}
      therapy:   {keywords: [therapy, cbt, dbt, ...],        color: "#4ade80"}
      risk_factor: {keywords: [trauma, abuse, stress, ...],   color: "#ffa500"}
      outcome:   {keywords: [suicide, relapse, recovery, ...], color: "#9b59b6"}
      population: {keywords: [adolescent, child, veteran, ...], color: "#3498db"}

  diabetes:
    search_terms: [type 2 diabetes management, insulin resistance, ...]
    entity_categories:
      condition: {keywords: [diabetes, hyperglycemia, ...],   color: "#ff6b9d"}
      treatment: {keywords: [insulin, metformin, ...],        color: "#4ade80"}
      # ...

  cancer:
    search_terms: [breast cancer treatment, chemotherapy efficacy, ...]
    entity_categories:
      cancer_type: {keywords: [breast cancer, lung cancer, ...], color: "#ff6b9d"}
      # ...
```

To switch domains, change the `default_domain` field and re-run the pipeline. The entire system — data collection, NLP extraction, entity classification, graph construction, and 3D visualization colors — automatically adapts.

---

## 🎓 **Tech Stack**

### AI/ML
| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.0+ | Deep learning framework, GCN encoder, HybridDecoder |
| **PyTorch Geometric** | 2.0+ | Graph neural network layers (GCNConv), link splitting |
| **Transformers (HuggingFace)** | — | SciBERT tokenizer and model loading |
| **SentenceTransformers** | — | Efficient [CLS] token extraction for 768-D embeddings |
| **Node2Vec** | — | Biased random walks + Word2Vec for topology embeddings |
| **NetworkX** | — | Graph construction, shortest paths, neighbor queries |
| **scikit-learn** | — | Baseline models (LR, RF), GridSearchCV, ROC-AUC |
| **XGBoost** | — | Gradient boosting baseline for link prediction |

### NLP
| Library | Purpose |
|---------|---------|
| **spaCy** | Core NLP pipeline (tokenization, sentence splitting) |
| **scispaCy** | Biomedical NLP extension for spaCy |
| **en_core_sci_sm** | Broad scientific NER model |
| **en_ner_bc5cdr_md** | Disease/Chemical NER from BioCreative V CDR |
| **en_core_web_sm** | Sentence boundary detection for relation extraction |
| **negspacy** (optional) | Negation detection using NegEx algorithm |

### Data & Web
| Technology | Purpose |
|-----------|---------|
| **SQLite** | Zero-configuration local database (papers, entities, relations) |
| **Flask** + **Flask-CORS** | REST API backend serving predictions and static frontend |
| **HTML/CSS/JS** | Vanilla SPA — no framework, no build step, no dependencies |
| **Plotly.js** | WebGL-accelerated 3D network graph visualization |

---

## ⚠️ **Limitations**

Understanding limitations is as important as understanding capabilities. These are known constraints of the current system:

| Limitation | Impact | Potential Mitigation |
|-----------|--------|---------------------|
| **Graph sparsity** | 872 papers → only 2.24% density. Many nodes have few connections, limiting topological signal. | Scale to 5,000–10,000+ papers using additional data sources |
| **Negation blindness** | "CBT is *not* effective for PTSD" still creates a CBT-PTSD edge because co-occurrence doesn't capture semantic direction | Integrate `negspacy` (already supported in code but optional) or use transformer-based relation extraction (e.g., REBEL) |
| **English only** | Ignores non-English research literature, biasing the knowledge graph toward English-language studies | Add multilingual SciBERT variants or translate abstracts |
| **Offline pipeline** | No real-time updates when new papers are published — requires re-running the entire pipeline | Build incremental update pipeline with periodic API polling |
| **Single domain tested** | Only Mental Health is fully demonstrated (Diabetes and Cancer configs exist but haven't been fully evaluated) | Validate on additional domains and publish comparative results |
| **Co-occurrence ≠ Causation** | Sentence-level co-occurrence doesn't distinguish between "treatment X cures disease Y" and "treatment X is unrelated to disease Y" | Use relation classification models to label edge types (treats, causes, correlates_with) |
| **Static graph** | The graph doesn't capture temporal dynamics (e.g., which connections are emerging trends vs. established knowledge) | Add temporal edge attributes (publication year) and train temporal GNNs |

---

## 🔮 **Future Work**

| Enhancement | Description | Benefit |
|------------|-------------|---------|
| **Transformer-based relation extraction** | Replace co-occurrence with models like REBEL or BioRE that classify relation types | More precise, labeled edges (treats, causes, inhibits) |
| **Graph Attention Networks (GAT)** | Replace GCN with GAT to learn attention weights per neighbor, making aggregation adaptive | Better performance on heterogeneous neighborhoods; attention weights serve as built-in explainability |
| **Integration with UMLS/DisGeNET** | Seed the graph with established biomedical knowledge bases before adding literature-mined edges | Dramatically denser graph with expert-curated relationships |
| **Real-time paper ingestion** | Periodically poll APIs for new papers and incrementally update the graph | Always up-to-date knowledge graph |
| **Multi-lingual support** | Process non-English abstracts using multilingual transformers | Broader literature coverage, fewer cultural biases |
| **Temporal GNN** | Add time-aware edge features and use TGNN architectures | Distinguish emerging connections from established ones |
| **User feedback loop** | Let domain experts confirm/reject predictions, fine-tuning the model | Human-in-the-loop learning improves accuracy over time |

---

## 📚 **References**

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| Semi-Supervised Classification with Graph Convolutional Networks | Kipf & Welling | ICLR 2017 | Foundation of the GCN architecture used in our encoder |
| node2vec: Scalable Feature Learning for Networks | Grover & Leskovec | KDD 2016 | Algorithm for our topology-aware embeddings |
| SciBERT: A Pretrained Language Model for Scientific Text | Beltagy et al. | EMNLP 2019 | Pre-trained transformer model for our 768-D semantic embeddings |
| Grad-CAM: Visual Explanations from Deep Networks | Selvaraju et al. | ICCV 2017 | Foundation for our gradient-based GNN attribution method |
| GNNExplainer: Generating Explanations for Graph Neural Networks | Ying et al. | NeurIPS 2019 | Alternative XAI method (not used due to speed constraints) |
| Explainability Methods for Graph Convolutional Neural Networks | Pope et al. | CVPR 2019 | Survey of GNN explanation methods that validates gradient-based approaches |
| The Link Prediction Problem for Social Networks | Liben-Nowell & Kleinberg | 2003 | Classic work on common neighbor-based link prediction (used in our explainability panels) |
| Batch Normalization: Accelerating Deep Network Training | Ioffe & Szegedy | ICML 2015 | Technique used in our GCN encoder for training stability |

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
  <sub>Strict Evaluation • Gradient-Based GNN Attribution • 896-D Hybrid Embeddings • Fully Transparent • Open Source</sub>
</p>

