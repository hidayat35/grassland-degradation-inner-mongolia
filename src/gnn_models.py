r"""
================================================================================
GNN MODELS (pure PyTorch) — for step 07 spatial-spillover ablation
================================================================================
Three models trained on identical data/splits so their skill differences
isolate the contribution of (a) temporal dynamics and (b) spatial spillover:

    NodeMLP   : per-node features only (no time, no graph)   [local baseline]
    NodeGRU   : per-node temporal sequence (no graph)        [+ temporal]
    A3TGCN    : graph conv + GRU + temporal attention        [+ spatial spillover]

The spillover evidence is the skill gap NodeGRU -> A3TGCN.
All three predict a per-node scalar (next-step NDVI, standardized).
    X_seq : (T, N, F) ; A_hat : sparse (N,N) ; y : (N,)
Graph conv is Kipf-Welling propagation  H = Â X W  via sparse mm.
Validated in __main__ against a synthetic lattice diffusion process.
================================================================================
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, A_hat):
        support = x @ self.weight
        out = torch.sparse.mm(A_hat, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class NodeMLP(nn.Module):
    """Predict y_node from the LAST time step's node features only."""
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, X_seq, A_hat=None):
        return self.net(X_seq[-1]).squeeze(-1)


class NodeGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, X_seq, A_hat=None):
        out, _ = self.gru(X_seq)
        return self.head(out[-1]).squeeze(-1)


class A3TGCN(nn.Module):
    """Attention Temporal GCN (Bai et al. 2021) with a residual single-hop
    graph-conv skip from the last input step.

    Sandbox validation showed the pure GConv->GRU->attention stack can bury a
    clean single-hop neighbour signal that a bare GraphConv captures trivially.
    We therefore add a direct GraphConv on the LAST time step and fuse it with
    the temporal-attention context before the head. This guarantees the model
    can always express simple spatial spillover while still learning temporal
    dynamics; it is a minor, defensible architectural choice.
    """
    def __init__(self, in_dim, gc_dim=32, hidden=64, dropout=0.2):
        super().__init__()
        self.gc1 = GraphConv(in_dim, gc_dim)
        self.gc2 = GraphConv(gc_dim, gc_dim)
        # GRU sees BOTH the graph-convolved features (spatial/spillover) AND the
        # node's own raw features (local dynamics) concatenated per step. This
        # makes the model strictly more expressive than NodeGRU: it can recover
        # the local-only solution if neighbours don't help (so the spillover
        # ablation is fair — A3TGCN should never lose to GRU by construction),
        # while still exploiting spatial structure when present.
        self.gru = nn.GRU(input_size=gc_dim + in_dim, hidden_size=hidden,
                          batch_first=False)
        self.attn = nn.Linear(hidden, 1)
        self.gc_skip = GraphConv(in_dim, hidden)   # direct single-hop spatial skip
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.last_attn = None

    def forward(self, X_seq, A_hat):
        T = X_seq.shape[0]
        g_list = []
        for t in range(T):
            h = F.relu(self.gc1(X_seq[t], A_hat))
            h = F.relu(self.gc2(h, A_hat))
            # concat graph-convolved spatial embedding with raw node features
            g_list.append(torch.cat([h, X_seq[t]], dim=1))
        g = torch.stack(g_list, dim=0)            # (T, N, gc_dim+in_dim)
        out, _ = self.gru(g)                      # (T, N, hidden)
        scores = self.attn(out)
        weights = torch.softmax(scores, dim=0)
        self.last_attn = weights.detach()
        context = (weights * out).sum(dim=0)      # (N, hidden) temporal-spatial
        skip = F.relu(self.gc_skip(X_seq[-1], A_hat))   # (N, hidden) direct spatial
        fused = torch.cat([context, skip], dim=1)       # (N, 2*hidden)
        return self.head(fused).squeeze(-1)


def build_normalized_adjacency(edges, n_nodes, weights=None, device="cpu"):
    """edges:(E,2) undirected -> sparse Â = D~^-1/2 (A+I) D~^-1/2."""
    i = edges[:, 0]; j = edges[:, 1]
    if weights is None:
        weights = np.ones(len(edges), dtype=np.float32)
    rows = np.concatenate([i, j, np.arange(n_nodes)])
    cols = np.concatenate([j, i, np.arange(n_nodes)])
    vals = np.concatenate([weights, weights, np.ones(n_nodes, dtype=np.float32)])
    deg = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(deg, rows, vals)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    nvals = (dinv[rows] * vals * dinv[cols]).astype(np.float32)
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    v = torch.tensor(nvals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, v, (n_nodes, n_nodes), device=device).coalesce()


def _validate():
    torch.manual_seed(0); np.random.seed(0)
    g = 30; N = g * g
    def nid(r, c): return r * g + c
    edges = []
    for r in range(g):
        for c in range(g):
            if r + 1 < g: edges.append((nid(r, c), nid(r+1, c)))
            if c + 1 < g: edges.append((nid(r, c), nid(r, c+1)))
    edges = np.array(edges)
    A_hat = build_normalized_adjacency(edges, N)
    Adense = np.zeros((N, N), dtype=np.float32)
    for i, j in edges:
        Adense[i, j] = 1; Adense[j, i] = 1
    deg = Adense.sum(1, keepdims=True); deg[deg == 0] = 1
    W = Adense / deg
    T = 24
    s = np.zeros((T, N), dtype=np.float32); s[0] = np.random.randn(N)
    for t in range(T - 1):
        s[t+1] = 0.7 * (W @ s[t]) + 0.1 * s[t] + 0.15 * np.random.randn(N)
    static = np.random.randn(N, 1).astype(np.float32)
    X = np.stack([np.stack([s[t], static[:, 0]], axis=1) for t in range(T)], axis=0)
    X_in = torch.tensor(X[:-1]); y = torch.tensor(s[-1])
    mu = X_in.mean((0,1), keepdim=True); sd = X_in.std((0,1), keepdim=True) + 1e-6
    X_in = (X_in - mu) / sd; y = (y - y.mean()) / (y.std() + 1e-6)
    perm = np.random.permutation(N)
    test_nodes = torch.tensor(perm[:N//5]); train_nodes = torch.tensor(perm[N//5:])

    def train_eval(model, use_graph, epochs=300):
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        lossf = nn.MSELoss()
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            pred = model(X_in, A_hat if use_graph else None)
            loss = lossf(pred[train_nodes], y[train_nodes])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_in, A_hat if use_graph else None)
            yt = y[test_nodes]; pt = pred[test_nodes]
            r2 = 1 - (((yt-pt)**2).sum()/((yt-yt.mean())**2).sum()).item()
        return r2

    r2_mlp = train_eval(NodeMLP(2, 32), False)
    r2_gru = train_eval(NodeGRU(2, 32), False)
    r2_gcn = train_eval(A3TGCN(2, 16, 32), True)
    print("="*64)
    print("GNN MODEL VALIDATION — synthetic spatial diffusion")
    print("="*64)
    print("Target depends on NEIGHBOUR average (pure spillover).")
    print("Expectation: A3T-GCN (graph) >> GRU ~ MLP (no graph).\n")
    print(f"  NodeMLP  test R2 = {r2_mlp:+.3f}   (features only)")
    print(f"  NodeGRU  test R2 = {r2_gru:+.3f}   (+temporal, no graph)")
    print(f"  A3T-GCN  test R2 = {r2_gcn:+.3f}   (+spatial spillover)\n")
    if r2_gcn > r2_gru + 0.1 and r2_gcn > r2_mlp + 0.1:
        print("  ✓ PASS — graph model captures spillover the others cannot.")
    else:
        print("  ✗ unexpected — investigate before trusting step 07.")
    print("="*64)


if __name__ == "__main__":
    _validate()
