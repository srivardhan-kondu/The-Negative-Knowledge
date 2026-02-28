"""
FINAL OPTIMIZED APPROACH: Bilinear + GCN Ensemble

Key insight: The 72% ceiling isn't from the architecture — it's from *how we score edges*.

SciBERT embeddings are trained for sentence similarity, not link prediction.
The key breakthrough is to train a LEARNED SIMILARITY METRIC (bilinear transform)
that adapts the pretrained embedding space to the task of predicting missing links.

This is called a "Bilinear Decoder" and is one of the most effective approaches for
knowledge graph completion. Instead of:
    score(u,v) = z_u • z_v  (raw dot product)
We use:
    score(u,v) = z_u^T × W × z_v  (learned bilinear transformation)

Where W is a learnable matrix that projects the embedding space to be maximally
discriminative for link prediction.

We also run multiple random seeds and pick the best run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import warnings
import copy

splits = torch.load("data/pyg_graph_splits.pt", weights_only=False)
train_data = splits["train_data"]
val_data = splits["val_data"]
test_data = splits["test_data"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
in_dim = train_data.x.shape[1]


class GCNEncoder(nn.Module):
    """Lightweight GCN that enriches node features with local graph context."""
    def __init__(self, in_dim, h, out_dim, dropout=0.5):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_dim)
        # Linear projection to reduce dim before convolution (much faster)
        self.proj = nn.Linear(in_dim, h)
        self.conv1 = GCNConv(h, h)
        self.bn1 = nn.BatchNorm1d(h)
        self.conv2 = GCNConv(h, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.bn_in(x)
        x = F.relu(self.proj(x))                         # Project to h
        res = x[:, :min(x.size(1), 64)]                   # Save residual (limited)
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, self.dropout, self.training)
        x = self.bn2(self.conv2(x, edge_index, edge_weight))
        return x


class BilinearDecoder(nn.Module):
    """
    Bilinear decoder: score(u,v) = z_u^T * W * z_v + b
    The learnable matrix W transforms the embedding space to be optimal
    for the specific link prediction task (structural + semantic alignment).
    """
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z, edge_index):
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        # z_src (N, dim) × W (dim, dim) = z_src_transformed (N, dim)
        z_src_transformed = z_src @ self.W
        # Dot product per edge pair
        score = (z_src_transformed * z_dst).sum(dim=1) + self.bias
        return score


class HybridDecoder(nn.Module):
    """
    Combines Bilinear transform with MLP for the best of both worlds.
    """
    def __init__(self, dim):
        super().__init__()
        self.bilinear = BilinearDecoder(dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, z, edge_index):
        bilinear_score = self.bilinear(z, edge_index)
        
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        mlp_score = self.mlp(torch.cat([z_src, z_dst, z_src * z_dst], dim=1)).squeeze()
        
        return bilinear_score + mlp_score


def run_trial(seed):
    torch.manual_seed(seed)
    
    model = GCNEncoder(in_dim, h=128, out_dim=64, dropout=0.45).to(device)
    decoder = HybridDecoder(64).to(device)
    
    params = list(model.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=0.003, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=150, T_mult=1, eta_min=5e-6)
    
    def decode(z, ei):
        return decoder(z, ei)
    
    best_auc = 0.0
    best_state = None
    patience = 80
    p_ctr = 0
    SMOOTH = 0.1

    for epoch in range(600):
        model.train(); decoder.train()
        opt.zero_grad()

        z = model(train_data.x, train_data.edge_index, train_data.edge_weight)
        pos = decode(z, train_data.edge_index)
        pos_lbl = torch.full((pos.size(0),), 1.0 - SMOOTH, device=device)

        neg_ei = negative_sampling(train_data.edge_index, train_data.num_nodes,
                                    num_neg_samples=train_data.edge_index.size(1))

        # 25% hard negatives from 2-hop
        adj_v = torch.ones(train_data.edge_index.size(1), device=device)
        adj = torch.sparse_coo_tensor(train_data.edge_index, adj_v,
                                      (train_data.num_nodes, train_data.num_nodes))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row, col = torch.sparse.mm(adj, adj).coalesce().indices()
        hm = row != col
        hc = torch.stack([row[hm], col[hm]])
        if hc.size(1) > 0:
            n_hard = int(train_data.edge_index.size(1) * 0.25)
            perm = torch.randperm(hc.size(1), device=device)[:n_hard]
            neg_ei = torch.cat([neg_ei[:, :neg_ei.size(1) - n_hard], hc[:, perm]], dim=1)

        neg = decode(z, neg_ei)
        neg_lbl = torch.full((neg.size(0),), SMOOTH, device=device)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]), torch.cat([pos_lbl, neg_lbl])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        model.eval(); decoder.eval()
        with torch.no_grad():
            z_v = model(val_data.x, train_data.edge_index, train_data.edge_weight)
            vp = decode(z_v, val_data.edge_label_index)
            vn_ei = negative_sampling(val_data.edge_label_index, val_data.num_nodes,
                                       num_neg_samples=val_data.edge_label_index.size(1))
            vn = decode(z_v, vn_ei)
            v_scores = torch.sigmoid(torch.cat([vp, vn])).cpu().numpy()
            v_true = [1]*len(vp) + [0]*len(vn)
            val_auc = roc_auc_score(v_true, v_scores)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {'model': copy.deepcopy(model.state_dict()),
                          'decoder': copy.deepcopy(decoder.state_dict())}
            p_ctr = 0
        else:
            p_ctr += 1

        if epoch % 100 == 0:
            print(f"  [Seed {seed}] Epoch {epoch:03d} | Loss {loss:.4f} | Val AUC {val_auc:.4f} | Best {best_auc:.4f}")

        if p_ctr >= patience:
            break

    return best_auc, best_state, model, decoder


# Run 3 random seeds and pick the best
print(f"GCNEncoder({in_dim}→128→64) + HybridDecoder (Bilinear + MLP)")
print(f"Running 3 random seeds to find best initialization...")
print("=" * 70)

best_overall_auc = 0.0
best_overall_state = None

for seed in [42, 123, 999]:
    print(f"\n--- Seed {seed} ---")
    val_auc, state, model_ref, dec_ref = run_trial(seed)
    print(f"  → Best Val AUC: {val_auc:.4f}")
    if val_auc > best_overall_auc:
        best_overall_auc = val_auc
        best_overall_state = state

# Final test evaluation
print(f"\n{'='*70}")
print(f"Best overall Val AUC: {best_overall_auc:.4f}. Evaluating on test set...")

# Re-instantiate model and decoder to load state
model = GCNEncoder(in_dim, h=128, out_dim=64, dropout=0.45).to(device)
decoder = HybridDecoder(64).to(device)
model.load_state_dict(best_overall_state['model'])
decoder.load_state_dict(best_overall_state['decoder'])

model.eval(); decoder.eval()
with torch.no_grad():
    z = model(test_data.x, train_data.edge_index, train_data.edge_weight)
    pos_out = torch.sigmoid(decoder(z, test_data.edge_label_index)).cpu()
    neg_ei = negative_sampling(test_data.edge_label_index, test_data.num_nodes,
                                num_neg_samples=test_data.edge_label_index.size(1))
    neg_out = torch.sigmoid(decoder(z, neg_ei)).cpu()
    auc = roc_auc_score([1]*len(pos_out)+[0]*len(neg_out),
                        list(pos_out.numpy())+list(neg_out.numpy()))

print(f"\n{'='*60}")
print(f"  🎯 Test GCN ROC-AUC: {auc:.4f} ({auc*100:.1f}%)")
print(f"{'='*60}")

torch.save({'model': model.state_dict(), 'decoder': decoder.state_dict()}, "data/gnn_model.pt")
print("Model saved.")
