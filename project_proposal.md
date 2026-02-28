# PROJECT PROPOSAL

**Discovering Negative Knowledge in Scientific Literature: An AI-Driven Approach to Identifying Research Gaps Using Graph Neural Networks**

---

## ABSTRACT

The exponential growth of scientific publications has created an information paradox: while we have more knowledge than ever, identifying what we *don't* know—the "negative knowledge"—has become increasingly challenging. This project proposes to develop an AI-driven system that will automatically discover under-researched connections in mental health literature by leveraging Graph Neural Networks (GNNs) and knowledge graph analysis.

The proposed system will collect research papers from multiple academic databases (Semantic Scholar, arXiv, PubMed), extract biomedical entities using advanced NLP techniques (scispaCy), construct knowledge graphs representing concept relationships, and employ Graph Convolutional Networks to predict missing but plausible connections. The target is to achieve >95% prediction accuracy while analyzing 500+ papers, extracting 600+ mental health concepts, and identifying 20-50 high-confidence research gaps that represent potential novel research directions.

This project addresses a critical gap in research methodology by transforming implicit knowledge discovery from a manual, serendipitous process into a systematic, AI-guided approach. By providing researchers with an interactive visualization platform featuring complete AI transparency, the proposed system will democratize access to research gap analysis and has the potential to accelerate scientific discovery in mental health and other medical domains.

**Keywords**: Graph Neural Networks, Knowledge Graphs, Research Gap Discovery, Negative Knowledge, Link Prediction, Biomedical NLP, AI Transparency

---

## 1. INTRODUCTION

### 1.1 Background and Motivation

The scientific community produces over 3 million research papers annually, with mental health research alone contributing 50,000+ publications per year. This information explosion creates several critical challenges:

1. **Information Overload**: Researchers cannot comprehensively review all relevant literature
2. **Hidden Connections**: Valuable interdisciplinary connections remain undiscovered
3. **Research Inefficiency**: Duplicate studies and missed opportunities waste resources
4. **Knowledge Fragmentation**: Insights scattered across publications don't synthesize

Traditional literature review methods are manual, time-intensive, and subject to individual cognitive limitations and biases. While existing tools help researchers *find* what is known, no systematic approach exists to identify what *should* be researched but hasn't been—the "negative knowledge."

### 1.2 Problem Statement

**How can we systematically discover under-researched connections and research gaps in scientific literature using AI, particularly in the mental health domain?**

Specific challenges to address:
- Identifying missing connections between established concepts
- Quantifying confidence in predicted relationships
- Ensuring AI transparency and interpretability
- Scaling across multiple medical domains
- Providing actionable insights for researchers

### 1.3 Research Objectives

**Primary Objective:**  
To design and develop an AI-powered system that automatically discovers under-researched connections (negative knowledge) in scientific literature using Graph Neural Networks and knowledge graph analysis.

**Specific Objectives:**
1. Design a multi-source data aggregation pipeline for collecting 500+ research papers
2. Develop a biomedical NLP pipeline for extracting and classifying 600+ mental health concepts
3. Construct a knowledge graph representing 2,000+ documented relationships
4. Implement and train a Graph Convolutional Network achieving >95% link prediction accuracy
5. Generate 20-50 high-confidence research gap predictions with quantified confidence scores
6. Create an interactive 3D visualization platform with complete AI transparency
7. Validate the approach across multiple medical domains

---

## 2. LITERATURE SURVEY

### 2.1 Knowledge Graph Construction for Scientific Literature

**1. DBpedia: A Nucleus for a Web of Open Data**
- Authors: Auer, S., et al.
- Year: 2007
- Link: https://doi.org/10.1007/978-3-540-76298-0_52
- **Key Contribution**: Pioneered automated knowledge base construction from Wikipedia
- **Limitation**: Focuses on general knowledge, not domain-specific scientific concepts

**2. YAGO: A Core of Semantic Knowledge**
- Authors: Rebele, T., et al.
- Year: 2016
- Link: https://doi.org/10.1016/j.artint.2014.05.002
- **Key Contribution**: High-quality knowledge base with temporal information
- **Limitation**: Manual curation required; limited to explicit statements

**3. Freebase: A Collaboratively Created Graph Database**
- Authors: Bollacker, K., et al.
- Year: 2008
- Link: https://doi.org/10.1145/1376616.1376746
- **Key Contribution**: Collaborative knowledge base construction
- **Limitation**: No predictive capabilities for missing information

### 2.2 Biomedical NLP and Entity Extraction

**4. BioBERT: Pre-trained Biomedical Language Representation Model**
- Authors: Lee, J., et al.
- Year: 2020
- Link: https://doi.org/10.1093/bioinformatics/btz682
- **Key Contribution**: Domain-adapted BERT for biomedical text
- **Limitation**: Entity extraction only; doesn't discover relationships or gaps

**5. ScispaCy: Fast and Robust Models for Biomedical NLP**
- Authors: Neumann, M., et al.
- Year: 2019
- Link: https://arxiv.org/abs/1902.07669
- **Key Contribution**: Efficient biomedical NER models
- **Limitation**: No knowledge graph integration or gap discovery

**6. PubMedBERT: Domain-Specific Language Model**
- Authors: Gu, Y., et al.
- Year: 2021
- Link: https://arxiv.org/abs/2007.15779
- **Key Contribution**: Improved performance on biomedical tasks
- **Limitation**: Lacks relationship prediction capabilities

### 2.3 Link Prediction and Graph Neural Networks

**7. node2vec: Scalable Feature Learning for Networks**
- Authors: Grover, A., & Leskovec, J.
- Year: 2016
- Link: https://doi.org/10.1145/2939672.2939754
- **Key Contribution**: Efficient graph embedding technique
- **Application**: Used for social networks, citation graphs
- **Gap**: Not adapted for scientific knowledge discovery

**8. Semi-Supervised Classification with Graph Convolutional Networks**
- Authors: Kipf, T. N., & Welling, M.
- Year: 2017
- Link: https://arxiv.org/abs/1609.02907
- **Key Contribution**: GCN architecture for graph-structured data
- **Gap**: Limited application to research gap identification

**9. Link Prediction Based on Graph Neural Networks**
- Authors: Zhang, M., & Chen, Y.
- Year: 2018
- Link: https://arxiv.org/abs/1802.09691
- **Key Contribution**: State-of-the-art link prediction methods
- **Gap**: Not tested on scientific literature analysis

**10. Graph Attention Networks**
- Authors: Veličković, P., et al.
- Year: 2018
- Link: https://arxiv.org/abs/1710.10903
- **Key Contribution**: Attention mechanism for graphs
- **Gap**: No domain-specific adaptations for medical research

### 2.4 Research Gap Identification Methods

**11. Bibliometric Analysis for Research Gap Identification**
- Authors: Hicks, D., et al.
- Year: 2018
- Link: https://doi.org/10.1093/reseval/rvy018
- **Key Contribution**: Framework for identifying research gaps
- **Limitation**: Manual expert analysis required; not scalable

**12. Automated Research Topic Generation**
- Authors: Wang, K., et al.
- Year: 2019
- Link: https://doi.org/10.1016/j.ipm.2019.102067
- **Key Contribution**: Keyword-based gap analysis
- **Limitation**: Misses semantic connections; no confidence metrics

**13. Literature-Based Discovery Methods**
- Authors: Swanson, D. R., & Smalheiser, N. R.
- Year: 1997
- Link: https://doi.org/10.1136/jamia.1997.0040283
- **Key Contribution**: ABC model for hidden knowledge discovery
- **Limitation**: Limited to co-occurrence; no deep learning

### 2.5 AI Transparency and Explainability

**14. Explainable AI: A Review**
- Authors: Adadi, A., & Berrada, M.
- Year: 2018
- Link: https://doi.org/10.1109/ACCESS.2018.2870052
- **Key Contribution**: Framework for interpretable AI
- **Gap**: Not applied to research gap discovery systems

**15. Attention is All You Need**
- Authors: Vaswani, A., et al.
- Year: 2017
- Link: https://arxiv.org/abs/1706.03762
- **Key Contribution**: Transformer architecture with interpretability
- **Gap**: Not designed for graph-structured scientific data

---

## 3. RESEARCH GAPS ANALYSIS

### 3.1 Summary of Identified Gaps

Based on comprehensive literature review, the following critical gaps exist in current research:

| **Gap ID** | **Research Gap** | **Current State** | **Proposed Solution** | **Impact** |
|------------|------------------|-------------------|----------------------|------------|
| **GAP-1** | **Lack of Automated Research Gap Discovery** | Manual expert analysis required; time-intensive and subjective | Design fully automated pipeline using GNN for systematic gap identification | High - Reduces discovery time from weeks to hours |
| **GAP-2** | **No Domain-Specific GNN for Scientific Literature** | GNNs used for social networks & citations; not adapted for medical research | Develop domain-adapted GCN with biomedical entity embeddings | High - First application of GNN to medical gap discovery |
| **GAP-3** | **Absence of Multi-Source Data Integration** | Most systems rely on single database (PubMed or Semantic Scholar) | Implement multi-source aggregation (3+ databases) for comprehensive coverage | Medium - Improves data diversity and coverage |
| **GAP-4** | **Missing AI Transparency in Research Tools** | Existing AI systems operate as "black boxes"; researchers cannot validate reasoning | Create transparency framework showing model architecture, metrics, and confidence scores | High - Builds trust and enables validation |
| **GAP-5** | **No Interactive Visualization for Knowledge Gaps** | Static reports and tables; limited exploration capabilities | Develop 3D interactive knowledge graph with real-time manipulation | Medium - Enhances user experience and exploration |
| **GAP-6** | **Inability to Quantify Research Gap Confidence** | Binary suggestions (yes/no); no probabilistic assessment | Implement probabilistic link prediction with 0-100% confidence scores | High - Enables priority-based research planning |
| **GAP-7** | **Limited Cross-Domain Transferability** | Systems designed for specific domains; cannot generalize | Design domain-agnostic architecture with configuration-based adaptation | Medium - Enables application to cancer, diabetes, etc. |

### 3.2 Significance of Addressing These Gaps

**Scientific Impact:**
- Transforms serendipitous discovery into systematic methodology
- Accelerates identification of novel research directions
- Reduces redundant research efforts

**Practical Impact:**
- Provides researchers with actionable gap lists ranked by confidence
- Enables evidence-based research planning
- Improves research resource allocation

**Methodological Impact:**
- Establishes new paradigm for "negative knowledge" discovery
- Combines NLP, graph theory, and deep learning innovatively
- Creates reproducible framework for gap analysis

---

## 4. PROPOSED METHODOLOGY

### 4.1 System Architecture (Conceptual Design)

```
┌─────────────────────────────────────────────┐
│         PROPOSED SYSTEM ARCHITECTURE        │
└─────────────────────────────────────────────┘

   ┌──────────────────┐
   │ DATA COLLECTION  │
   │  Multi-Source    │
   │  (Semantic       │
   │   Scholar,       │
   │   arXiv, PubMed) │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  NLP PIPELINE    │
   │  (ScispaCy +     │
   │   BioBERT)       │
   │  - Entity        │
   │    Extraction    │
   │  - Classification│
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ KNOWLEDGE GRAPH  │
   │ CONSTRUCTION     │
   │  (NetworkX)      │
   │  - Nodes:        │
   │    Concepts      │
   │  - Edges:        │
   │    Relations     │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ GRAPH EMBEDDINGS │
   │  (Node2Vec)      │
   │  64-dimensional  │
   │  representations │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │   GNN TRAINING   │
   │ (Graph Conv Net) │
   │  - Link          │
   │    Prediction    │
   │  - Target: >95%  │
   │    accuracy      │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  RESEARCH GAP    │
   │   PREDICTION     │
   │  - Rank missing  │
   │    connections   │
   │  - Confidence    │
   │    scoring       │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  VISUALIZATION   │
   │  Interactive 3D  │
   │  Knowledge Graph │
   │  + Transparency  │
   │    Panels        │
   └──────────────────┘
```

### 4.2 Proposed Implementation Phases

**Phase 1: Multi-Source Data Collection (Weeks 1-2)**
- Design API integration for Semantic Scholar, arXiv, PubMed
- Implement data scraping with rate limiting and error handling
- Target: Collect 500-700 mental health research papers
- Store in structured database with metadata

**Phase 2: NLP Processing Pipeline (Weeks 2-3)**
- Implement scispaCy and BioBERT for entity extraction
- Design rule-based classification for 5 categories:
  - Mental disorders
  - Therapies/treatments
  - Risk factors
  - Population groups
  - Outcomes
- Target: Extract 600+ unique concepts

**Phase 3: Relation Extraction (Weeks 3-4)**
- Implement co-occurrence analysis at sentence level
- Extract relationships between entities
- Filter low-confidence relations
- Target: 2,000+ documented relationships

**Phase 4: Knowledge Graph Construction (Week 4)**
- Design graph schema using NetworkX
- Implement graph building algorithm
- Compute centrality metrics
- Target: Connected graph with meaningful structure

**Phase 5: Graph Embedding Training (Week 5)**
- Implement Node2Vec algorithm
- Tune hyperparameters (dimensions, walk length, etc.)
- Generate 64-dimensional embeddings
- Validate with similarity checks

**Phase 6: GNN Model Development (Weeks 6-7)**
- Design Graph Convolutional Network architecture
- Implement training pipeline with PyTorch Geometric
- Use binary cross-entropy loss for link prediction
- Target: >95% ROC-AUC on validation set

**Phase 7: Research Gap Prediction (Weeks 7-8)**
- Generate candidate missing edges
- Score predictions using trained GNN
- Rank by confidence (0-100%)
- Select top 20-50 high-confidence gaps

**Phase 8: Interactive Visualization (Weeks 8-9)**
- Design 3D graph visualization with Plotly
- Implement interactivity (rotation, zoom, hover)
- Create transparency panels showing:
  - Model architecture and performance
  - Top predictions with confidence
  - Methodology explanation
- Ensure user-friendly interface

**Phase 9: Evaluation & Validation (Weeks 9-10)**
- Validate predictions against recent literature
- Conduct expert review with domain specialists
- Perform user study for usability
- Document results and limitations

**Phase 10: Documentation & Reporting (Weeks 10-12)**
- Prepare technical documentation
- Write research paper draft
- Create user guide and tutorials
- Present findings

### 4.3 Tools and Technologies (Proposed)

**Programming & Libraries:**
- Python 3.11+ for implementation
- SpaCy, scispaCy for NLP
- NetworkX for graph manipulation
- PyTorch + PyTorch Geometric for GNN
- Plotly for visualization
- Pandas, NumPy for data processing

**Data Sources:**
- Semantic Scholar Graph API
- arXiv API
- PubMed Entrez E-utilities

**Development Environment:**
- Git for version control
- Jupyter notebooks for experimentation
- Virtual environment for dependency management

---

## 5. EXPECTED OUTCOMES AND DELIVERABLES

### 5.1 Technical Deliverables

1. **Functional AI System**
   - Multi-source data collection module
   - NLP processing pipeline
   - Graph construction engine
   - Trained GNN model (>95% accuracy target)
   - Interactive visualization platform

2. **Research Outputs**
   - Knowledge graph of 600+ mental health concepts
   - List of 20-50 high-confidence research gaps
   - Quantitative evaluation metrics
   - Comparison with existing methods

3. **Documentation**
   - System architecture document
   - User manual and tutorials
   - API documentation
   - Research methodology paper

### 5.2 Academic Contributions

1. **Novel Methodology**: First systematic AI approach to "negative knowledge" discovery in medical literature
2. **Technical Innovation**: Domain-adapted GNN for scientific gap analysis
3. **Practical Tool**: Usable system for researchers
4. **Reproducible Research**: Open methodology and documentation

### 5.3 Success Criteria

| **Criterion** | **Target** | **Measurement Method** |
|---------------|------------|----------------------|
| Data Collection | 500+ papers from 3+ sources | Database query count |
| Entity Extraction | 600+ unique concepts | Entity table count |
| Graph Size | 2,000+ relationships | Edge count in graph |
| Model Accuracy | >95% ROC-AUC | Test set evaluation |
| Prediction Quality | 20+ high-confidence gaps | Confidence threshold >90% |
| Literature Validation | ≥70% novel predictions | Manual verification vs recent papers |
| Expert Validation | ≥4/5 average rating | Domain expert survey (5-point scale) |
| Usability Score | ≥70/100 | System Usability Scale (SUS) |

---

## 6. PROJECT TIMELINE

### Gantt Chart (12 Weeks)

| Week | Phase | Key Activities | Expected Output |
|------|-------|----------------|----------------|
| 1-2 | Data Collection | API design, data scraping, database setup | 500+ papers collected |
| 2-3 | NLP Processing | Entity extraction, classification | 600+ concepts extracted |
| 3-4 | Relation Extraction | Co-occurrence analysis, validation | 2,000+ relationships |
| 4 | Graph Building | NetworkX implementation, statistics | Knowledge graph file |
| 5 | Embedding Training | Node2Vec implementation | 64D embeddings |
| 6-7 | GNN Development | Model design, training, evaluation | Trained GNN model |
| 7-8 | Gap Prediction | Link prediction, ranking | Top 20-50 research gaps |
| 8-9 | Visualization | UI development, interactivity | 3D visualization tool |
| 9-10 | Evaluation | Metrics, expert review, user study | Validation report |
| 10-11 | Documentation | Technical docs, user guide | Complete documentation |
| 11-12 | Paper Writing | Research paper draft, presentation | Final deliverables |

**Critical Milestones:**
- Week 4: Knowledge graph completed
- Week 7: GNN model trained (>95% accuracy)
- Week 9: Visualization platform ready
- Week 10: Evaluation completed
- Week 12: Project delivery and presentation

---

## 7. RESOURCE REQUIREMENTS

### 7.1 Human Resources
- Student researcher (full-time, 12 weeks)
- Faculty advisor (2-3 hours/week for guidance)
- Domain expert (1-2 hours for validation)

### 7.2 Computational Resources
- Personal computer (Quad-core CPU, 8GB+ RAM)
- Internet connection for API access
- Optional: Google Colab for GPU acceleration (free tier)

### 7.3 Software Resources
- Python ecosystem (free, open-source)
- Academic API access (free tiers available)
- Visualization libraries (free)

### 7.4 Estimated Budget
- **₹0-5,000** (minimal cost)
  - Most resources freely available
  - Potential costs: Domain expert consultation, cloud computing (if needed)

---

## 8. RISK ANALYSIS AND MITIGATION

| **Risk** | **Probability** | **Impact** | **Mitigation Strategy** |
|----------|----------------|-----------|------------------------|
| API rate limits | Medium | Medium | Implement exponential backoff; use multiple APIs |
| Low NLP accuracy | Low | High | Use ensemble models; manual validation sample |
| Sparse knowledge graph | Medium | Medium | Collect more data; use graph augmentation |
| Computational constraints | Low | Medium | Optimize algorithms; use cloud resources if needed |
| Difficulty in expert validation | Medium | High | Partner with university research groups early |
| Model doesn't achieve target accuracy | Low | High | Try alternative architectures (GAT, GraphSAGE) |
| Limited generalization | Medium | Low | Design modular architecture from start |

---

## 9. FUTURE SCOPE

1. **Multi-Domain Expansion**: Extend to cancer, diabetes, cardiovascular research
2. **Temporal Analysis**: Track research trends and gap evolution over time
3. **Causal Inference**: Predict causal relationships, not just associations
4. **Researcher Networking**: Suggest collaboration opportunities based on gaps
5. **Automated Literature Review**: Generate systematic review sections
6. **Integration with Research Databases**: Direct API integration with institutional libraries

---

## 10. CONCLUSION

This project proposes to address a fundamental challenge in modern research: systematically identifying what we don't know. By combining multi-source data aggregation, advanced NLP, Graph Neural Networks, and interactive visualization, we aim to create a transformative tool for accelerating scientific discovery in mental health and beyond.

The project fills seven identified research gaps, contributes novel methodology, and will deliver practical impact. With a clear 12-week timeline, achievable objectives, and comprehensive evaluation plan, this project is well-positioned for successful completion and meaningful contribution to research methodology.

**The Goal**: Transform serendipitous discovery into systematic science, democratize access to research gap analysis, and accelerate progress in mental health research.

---

## REFERENCES

1. Auer, S., et al. (2007). DBpedia: A nucleus for a web of open data. *The Semantic Web*. https://doi.org/10.1007/978-3-540-76298-0_52

2. Rebele, T., et al. (2016). YAGO: A multilingual knowledge base. *Artificial Intelligence*. https://doi.org/10.1016/j.artint.2014.05.002

3. Bollacker, K., et al. (2008). Freebase: A collaboratively created graph database. *ACM SIGMOD*. https://doi.org/10.1145/1376616.1376746

4. Lee, J., et al. (2020). BioBERT: Pre-trained biomedical model. *Bioinformatics*. https://doi.org/10.1093/bioinformatics/btz682

5. Neumann, M., et al. (2019). ScispaCy: Biomedical NLP. *arXiv*. https://arxiv.org/abs/1902.07669

6. Gu, Y., et al. (2021). PubMedBERT. *arXiv*. https://arxiv.org/abs/2007.15779

7. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning. *KDD*. https://doi.org/10.1145/2939672.2939754

8. Kipf, T. N., & Welling, M. (2017). Graph convolutional networks. *ICLR*. https://arxiv.org/abs/1609.02907

9. Zhang, M., & Chen, Y. (2018). Link prediction with GNN. *NeurIPS*. https://arxiv.org/abs/1802.09691

10. Veličković, P., et al. (2018). Graph attention networks. *ICLR*. https://arxiv.org/abs/1710.10903

11. Hicks, D., et al. (2018). Bibliometric analysis for gaps. *Research Evaluation*. https://doi.org/10.1093/reseval/rvy018

12. Wang, K., et al. (2019). Automated topic generation. *Information Processing*. https://doi.org/10.1016/j.ipm.2019.102067

13. Swanson, D. R., & Smalheiser, N. R. (1997). Literature-based discovery. *JAMIA*. https://doi.org/10.1136/jamia.1997.0040283

14. Adadi, A., & Berrada, M. (2018). Explainable AI. *IEEE Access*. https://doi.org/10.1109/ACCESS.2018.2870052

15. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*. https://arxiv.org/abs/1706.03762

---

**Submitted by**: [Your Name]  
**Student ID**: [Your ID]  
**Department**: Computer Science / AI & ML  
**Academic Year**: 2025-2026  
**Project Type**: Final Year / Capstone Project  
**Expected Duration**: 12 weeks
