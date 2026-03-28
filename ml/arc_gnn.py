#!/usr/bin/env python3
"""
arc_gnn.py — GraphSAGE GNN for cross-resolution phase transition prediction.

Predicts parent corpus phase_transition_score from subclass cluster graphs.

Usage:
  python3 arc_gnn.py --corpus all                       # G06N all subclasses
  python3 arc_gnn.py --corpus G06N_3_quarterly          # single corpus
  python3 arc_gnn.py --parent-corpus H01L_quarterly     # H01L all subclasses
  python3 arc_gnn.py --parent-corpus H01L_quarterly --temporal  # + temporal

Graph structure per (corpus_id, period_start):
  Nodes   : non-junk clusters (10 features from clusters table)
  Edges   : cluster_edges (cluster_a ↔ cluster_b, connection_weight)
  Target  : parent_phase_transition_score from v_cross_resolution_period_stats

Bugs fixed vs original spec:
  - Node query uses clusters table directly; v_cluster_event_training had no
    period_start column (now added, but direct clusters query is still cleaner)
  - Edges use cluster_edges, not knn_edges (knn_edges connects chunks not
    clusters; columns are chunk_id/neighbor_id/distance, no chunk_id_a)
  - graph_ctx stored as [1, 4] so DataLoader collates to [batch_size, 4]
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import psycopg2


def _make_engine():
    pghost = os.environ.get("PGHOST", "/var/run/postgresql")
    pgdb   = os.environ.get("PGDATABASE", "arc_v4")
    pguser = os.environ.get("PGUSER", "jeff")
    if pghost.startswith("/"):
        # Unix socket: pass host as a query param, not in netloc
        url = f"postgresql+psycopg2://{pguser}@/{pgdb}?host={pghost}"
    else:
        url = f"postgresql+psycopg2://{pguser}@{pghost}/{pgdb}"
    return create_engine(url)


def _get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


engine = _make_engine()
MODELS_DIR = Path(__file__).parent / 'models'

G06N_SUBCLASS_CORPORA = [
    'G06N_3_quarterly',
    'G06N_5_quarterly',
    'G06N_7_quarterly',
    'G06N_10_quarterly',
    'G06N_20_quarterly',
]

H01L_SUBCLASS_CORPORA = [
    'H01L_21_quarterly',
    'H01L_22_quarterly',
    'H01L_23_quarterly',
    'H01L_24_quarterly',
    'H01L_25_quarterly',
]

# Backward-compat alias (G06N default)
SUBCLASS_CORPORA = G06N_SUBCLASS_CORPORA

SUBCLASS_CORPORA_BY_PARENT = {
    'G06N_quarterly': G06N_SUBCLASS_CORPORA,
    'H01L_quarterly': H01L_SUBCLASS_CORPORA,
}

NODE_FEATURE_COLS = [
    'size', 'cohesion', 'drift_magnitude', 'convergence_score',
    'mean_betweenness', 'persistence_score', 'elongation_ratio',
    'marginal_entropy_impact', 'boundary_pressure_rate', 'jerk',
]
IN_CHANNELS   = len(NODE_FEATURE_COLS)  # 10
GRAPH_CTX_DIM = 4                       # cross-resolution scalars


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def build_graphs(corpus_filter=None, parent_corpus='G06N_quarterly'):
    """
    Return list of PyG Data objects, one per (subclass_corpus, period).

    corpus_filter  : str or None — restrict to a single corpus_id.
    parent_corpus  : str — filter by parent_corpus_id in view.
    """
    clauses = ["v.target IS NOT NULL",
               "v.parent_corpus_id = %(parent)s"]
    params = {'parent': parent_corpus}

    if corpus_filter is not None:
        clauses.append("v.corpus_id = %(cid)s")
        params['cid'] = corpus_filter

    where_sql = " AND ".join(clauses)

    meta = pd.read_sql(f'''
        SELECT
            v.corpus_id,
            v.period_start,
            v.parent_corpus_id,
            COALESCE(v.subclass_pts, 0)                  AS subclass_pts,
            v.target                                     AS target,
            COALESCE(v.drift_divergence, 0)              AS drift_divergence,
            COALESCE(v.entropy_ratio, 1)                 AS entropy_ratio,
            COALESCE(v.modularity_ratio, 1)              AS modularity_ratio
        FROM v_cross_resolution_period_stats v
        WHERE {where_sql}
        ORDER BY v.corpus_id, v.period_start
    ''', engine, params=params)

    label = corpus_filter or f'all({parent_corpus})'
    print(f"  [{label}] Candidate periods: {len(meta)}")

    graphs = []
    skipped_small    = 0
    skipped_no_edges = 0

    for _, row in meta.iterrows():
        corpus_id    = row['corpus_id']
        period_start = (row['period_start'].strftime('%Y-%m-%d')
                        if hasattr(row['period_start'], 'strftime')
                        else str(row['period_start'])[:10])
        target = float(row['target'])

        # Node features from clusters table (NOT v_cluster_event_training —
        # that view is a flat projection with no period_start filter).
        nodes = pd.read_sql('''
            SELECT
                cluster_id,
                COALESCE(size, 0)::real                    AS size,
                COALESCE(cohesion, 0)::real                AS cohesion,
                COALESCE(drift_magnitude, 0)::real         AS drift_magnitude,
                COALESCE(convergence_score, 0)::real       AS convergence_score,
                COALESCE(mean_betweenness, 0)::real        AS mean_betweenness,
                COALESCE(persistence_score, 0)::real       AS persistence_score,
                COALESCE(elongation_ratio, 1.5)::real      AS elongation_ratio,
                COALESCE(marginal_entropy_impact, 0)::real AS marginal_entropy_impact,
                COALESCE(boundary_pressure_rate, 0)::real  AS boundary_pressure_rate,
                COALESCE(jerk, 0)::real                    AS jerk
            FROM pipe_clusters
            WHERE corpus_id = %s
              AND period_start = %s
              AND is_junk = FALSE
            ORDER BY cluster_id
        ''', engine, params=(corpus_id, period_start))

        if len(nodes) < 2:
            skipped_small += 1
            continue

        id_map = {int(cid): i for i, cid in enumerate(nodes['cluster_id'])}
        x = torch.tensor(
            nodes[NODE_FEATURE_COLS].values.astype(np.float32),
            dtype=torch.float,
        )

        # Edges from cluster_edges (cluster-level graph).
        # knn_edges is chunk-to-chunk and uses different column names.
        edges = pd.read_sql('''
            SELECT cluster_a, cluster_b,
                   COALESCE(connection_weight, 1.0)::real AS w
            FROM pipe_cluster_edges
            WHERE corpus_id = %s AND period_start = %s
        ''', engine, params=(corpus_id, period_start))

        src, dst, wts = [], [], []
        for _, e in edges.iterrows():
            a, b = int(e['cluster_a']), int(e['cluster_b'])
            if a in id_map and b in id_map:
                ia, ib = id_map[a], id_map[b]
                src += [ia, ib];  dst += [ib, ia]
                wts += [float(e['w'])] * 2

        if len(src) == 0:
            n = len(nodes)
            for i in range(n):
                for j in range(i + 1, n):
                    src += [i, j];  dst += [j, i];  wts += [1.0, 1.0]
            skipped_no_edges += 1

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(wts, dtype=torch.float).unsqueeze(1)

        # graph_ctx shape [1, 4]: DataLoader stacks to [batch_size, 4].
        # Shape [4] would concatenate to [4*batch_size] — wrong.
        graph_ctx = torch.tensor([[
            float(row['subclass_pts']),
            float(row['drift_divergence']),
            float(row['entropy_ratio']),
            float(row['modularity_ratio']),
        ]], dtype=torch.float)

        y = torch.tensor([target], dtype=torch.float)

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            graph_ctx=graph_ctx, y=y,
            corpus_id=corpus_id,
            period_start=period_start,
        )
        graphs.append(data)

    print(f"  [{label}] Built: {len(graphs)}  "
          f"skipped: {skipped_small} (<2 nodes), "
          f"{skipped_no_edges} fallback full-connect")
    return graphs


# ---------------------------------------------------------------------------
# Tabular baseline per corpus
# ---------------------------------------------------------------------------

def compute_tabular_baseline(corpus_filter=None, parent_corpus='G06N_quarterly'):
    """
    Pearson r between subclass phase_transition_score and parent
    phase_transition_score → R² = r².  One value per corpus filter.
    """
    clauses = ["v.target IS NOT NULL",
               "v.parent_corpus_id = %(parent)s"]
    params = {'parent': parent_corpus}

    if corpus_filter:
        clauses.append("v.corpus_id = %(cid)s")
        params['cid'] = corpus_filter

    where_sql = " AND ".join(clauses)

    df = pd.read_sql(f'''
        SELECT subclass_pts,
               target
        FROM v_cross_resolution_period_stats v
        WHERE {where_sql}
    ''', engine, params=params)

    if len(df) < 3:
        return 0.0
    r = float(np.corrcoef(df['subclass_pts'].fillna(0),
                          df['target'])[0, 1])
    return r ** 2  # R²


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PhaseTransitionGNN(torch.nn.Module):
    """3-layer GraphSAGE + global mean pool + cross-resolution context → scalar."""

    def __init__(self, in_channels=IN_CHANNELS,
                 hidden_channels=32,
                 graph_ctx_dim=GRAPH_CTX_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin   = torch.nn.Linear(hidden_channels + graph_ctx_dim, 1)

    def forward(self, x, edge_index, batch, graph_ctx):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.cat([x, graph_ctx], dim=1)
        return self.lin(x).squeeze(-1)


class TemporalCrossResGNN(torch.nn.Module):
    """
    Temporal variant: combines T, T-1, T-2 subclass graph embeddings
    (each with graph_ctx) for parent phase_transition_score prediction.
    """
    def __init__(self, in_channels=IN_CHANNELS, hidden=32,
                 graph_ctx_dim=GRAPH_CTX_DIM):
        super().__init__()
        # Shared encoder (same weights for all time steps)
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        # Combine 3 time-step embeddings + 3 graph_ctx vectors
        fused_dim  = hidden * 3 + graph_ctx_dim * 3
        self.temporal_lin = torch.nn.Linear(fused_dim, hidden)
        self.output_lin   = torch.nn.Linear(hidden, 1)

    def _embed(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)

    def forward(self, x0, ei0, b0, ctx0,
                      x1, ei1, b1, ctx1,
                      x2, ei2, b2, ctx2):
        h = torch.cat([
            self._embed(x0, ei0, b0), ctx0,
            self._embed(x1, ei1, b1), ctx1,
            self._embed(x2, ei2, b2), ctx2,
        ], dim=-1)
        return self.output_lin(F.relu(self.temporal_lin(h))).squeeze(-1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def r2_score(preds, targets):
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def evaluate_r2(model, graphs, device):
    """Run inference on a graph list and return R²."""
    if len(graphs) == 0:
        return 0.0
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, batch.graph_ctx)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    return r2_score(np.array(preds), np.array(targets))


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_gnn(model, graphs, device, feature_names):
    """
    Zero-ablation permutation importance for node features.

    For each feature i, set x[:, i] = 0 across all graphs and measure
    the drop in R² vs baseline.  Larger drop = more important feature.

    Returns list of (feature_name, importance) sorted descending.
    """
    baseline_r2 = evaluate_r2(model, graphs, device)
    importances = {}

    for i, fname in enumerate(feature_names):
        perturbed = []
        for g in graphs:
            g_copy = g.clone()
            g_copy.x = g_copy.x.clone()
            g_copy.x[:, i] = 0.0
            perturbed.append(g_copy)

        perturbed_r2 = evaluate_r2(model, perturbed, device)
        importances[fname] = baseline_r2 - perturbed_r2

    return sorted(importances.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Training — base cross-resolution GNN
# ---------------------------------------------------------------------------

def train_gnn(graphs, corpus_label='all', checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on: {device}")

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(graphs))
    split = int(0.8 * len(graphs))
    train_graphs = [graphs[i] for i in idx[:split]]
    test_graphs  = [graphs[i] for i in idx[split:]]
    print(f"  Train: {len(train_graphs)}  |  Test: {len(test_graphs)}")

    if len(test_graphs) < 4:
        print(f"  WARNING: only {len(test_graphs)} test graphs — R² unreliable")

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)

    model     = PhaseTransitionGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if checkpoint is None:
        checkpoint = str(MODELS_DIR / f'arc_gnn_{corpus_label}.pt')

    best_r2    = -999.0
    best_epoch = -1

    for epoch in range(200):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, batch.graph_ctx)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            r2 = evaluate_r2(model, test_graphs, device)
            if r2 > best_r2:
                best_r2    = r2
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            print(f"  Epoch {epoch:3d}: loss={total_loss:.4f}  "
                  f"test R²={r2:.3f}  (best={best_r2:.3f} @ ep{best_epoch})")

    # Load best weights for permutation importance
    model.load_state_dict(torch.load(checkpoint, weights_only=True))

    # Permutation importance on full graph set (more stable than test-only)
    print(f"\n  Computing permutation importance ({len(graphs)} graphs)...")
    perm_imp = permutation_importance_gnn(model, graphs, device, NODE_FEATURE_COLS)

    print(f"\n  {'─'*55}")
    print(f"  [{corpus_label}] Best test R²  : {best_r2:.3f}  (epoch {best_epoch})")
    print(f"  [{corpus_label}] Checkpoint    : {checkpoint}")
    print(f"\n  Permutation importance (R² drop when feature zeroed):")
    for fname, imp in perm_imp:
        bar = '█' * max(0, int(imp * 200))
        print(f"    {fname:30s}  {imp:+.4f}  {bar}")
    print(f"  {'─'*55}\n")

    return best_r2, perm_imp, model, device, test_graphs


# ---------------------------------------------------------------------------
# Training — temporal cross-resolution GNN
# ---------------------------------------------------------------------------

def build_temporal_triples(graphs):
    """
    Group graphs by (corpus_id, period_start) and build T, T-1, T-2 triples.
    Triples are within the same corpus_id (temporal evolution of one subclass).
    """
    # Group by corpus_id, sort by period_start within each group
    by_corpus = {}
    for g in graphs:
        cid = g.corpus_id
        by_corpus.setdefault(cid, []).append(g)

    triples = []
    for cid, cg in by_corpus.items():
        cg_sorted = sorted(cg, key=lambda g: g.period_start)
        by_ps = {g.period_start: g for g in cg_sorted}
        periods = sorted(by_ps.keys())
        for i in range(2, len(periods)):
            p0, p1, p2 = periods[i], periods[i-1], periods[i-2]
            triples.append((by_ps[p0], by_ps[p1], by_ps[p2]))

    return triples


def train_temporal_gnn(graphs, corpus_label='all', checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    triples = build_temporal_triples(graphs)
    print(f"  Temporal triples: {len(triples)}")

    if len(triples) < 10:
        print("  Too few triples — skipping temporal training.")
        return None

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(triples))
    split = int(0.8 * len(triples))
    train_triples = [triples[i] for i in idx[:split]]
    test_triples  = [triples[i] for i in idx[split:]]
    print(f"  Train: {len(train_triples)}  Test: {len(test_triples)}")

    if checkpoint is None:
        checkpoint = str(MODELS_DIR / f'arc_gnn_{corpus_label}_temporal.pt')

    model     = TemporalCrossResGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def batch_triples(chunk):
        def clean(g):
            return Data(x=g.x, edge_index=g.edge_index, y=g.y)

        b0 = Batch.from_data_list([clean(t[0]) for t in chunk]).to(device)
        b1 = Batch.from_data_list([clean(t[1]) for t in chunk]).to(device)
        b2 = Batch.from_data_list([clean(t[2]) for t in chunk]).to(device)
        # graph_ctx: [1, 4] per graph → stack to [batch_size, 4]
        ctx0 = torch.cat([t[0].graph_ctx for t in chunk], dim=0).to(device)
        ctx1 = torch.cat([t[1].graph_ctx for t in chunk], dim=0).to(device)
        ctx2 = torch.cat([t[2].graph_ctx for t in chunk], dim=0).to(device)
        y    = torch.tensor([t[0].y.item() for t in chunk],
                             dtype=torch.float).to(device)
        return b0, b1, b2, ctx0, ctx1, ctx2, y

    def eval_temporal(triple_list):
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for start in range(0, len(triple_list), 16):
                chunk = triple_list[start:start+16]
                b0, b1, b2, c0, c1, c2, y = batch_triples(chunk)
                pred = model(b0.x, b0.edge_index, b0.batch, c0,
                             b1.x, b1.edge_index, b1.batch, c1,
                             b2.x, b2.edge_index, b2.batch, c2)
                preds.extend(pred.cpu().numpy())
                tgts.extend(y.cpu().numpy())
        return r2_score(np.array(preds), np.array(tgts))

    best_r2, best_epoch = -999.0, -1
    for epoch in range(200):
        model.train()
        total_loss = 0.0
        shuffled = [train_triples[i] for i in
                    np.random.default_rng(epoch).permutation(len(train_triples))]
        for start in range(0, len(shuffled), 16):
            chunk = shuffled[start:start+16]
            b0, b1, b2, c0, c1, c2, y = batch_triples(chunk)
            optimizer.zero_grad()
            pred = model(b0.x, b0.edge_index, b0.batch, c0,
                         b1.x, b1.edge_index, b1.batch, c1,
                         b2.x, b2.edge_index, b2.batch, c2)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            r2 = eval_temporal(test_triples)
            if r2 > best_r2:
                best_r2, best_epoch = r2, epoch
                torch.save(model.state_dict(), checkpoint)
            print(f"  Epoch {epoch:3d}: loss={total_loss:.4f}  R²={r2:.3f}  "
                  f"(best={best_r2:.3f} @ ep{best_epoch})")

    print(f"\n  Temporal best R²: {best_r2:.3f} (epoch {best_epoch})")
    print(f"  Checkpoint: {checkpoint}")
    return best_r2


# ---------------------------------------------------------------------------
# Save result to findings
# ---------------------------------------------------------------------------

def save_finding(corpus_label, best_r2, baseline_r2, perm_imp, n_graphs,
                 parent_corpus='G06N_quarterly'):
    subclass_list = SUBCLASS_CORPORA_BY_PARENT.get(parent_corpus,
                                                    G06N_SUBCLASS_CORPORA)
    top3 = ', '.join(f[0] for f, _ in zip(perm_imp[:3], range(3)))
    body = (
        f"GraphSAGE GNN (3 layers, hidden=32) on {corpus_label} subclass cluster "
        f"graphs (nodes=clusters, edges=cluster_edges, graph_ctx=cross-resolution "
        f"scalars). Best test R²={best_r2:.3f} vs tabular R²={baseline_r2:.3f}. "
        f"GNN gain: {best_r2 - baseline_r2:+.3f}. "
        f"n_graphs={n_graphs}, 80/20 split. "
        f"Top-3 permutation-important node features: {top3}."
    )
    confidence = min(0.55 + max(best_r2, 0) * 0.35, 0.92)

    corpus_ids = (
        subclass_list
        if corpus_label in ('all', f'all({parent_corpus})')
        else [corpus_label, parent_corpus]
    )

    perm_dict  = {f: float(f'{v:.4f}') for f, v in perm_imp}
    cond_expr  = {
        'view':            'cluster_edges + v_cross_resolution_period_stats',
        'target':          'parent_phase_transition_score',
        'model':           'GraphSAGE 3-layer hidden=32',
        'corpus':          corpus_label,
        'parent_corpus':   parent_corpus,
        'baseline_r2':     round(baseline_r2, 4),
        'best_r2':         round(best_r2, 4),
        'scope':           'quarterly',
        'status':          'ready',
        'perm_importance': perm_dict,
    }

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sci_findings
                (type, title, body, confidence, status, corpus_ids,
                 condition_expression)
            VALUES (
                'law',
                %(title)s,
                %(body)s,
                %(conf)s,
                'active',
                %(cids)s,
                %(cexpr)s::jsonb
            )
        """, {
            'title': f'GNN Phase Transition Predictor — {corpus_label}',
            'body':   body,
            'conf':   confidence,
            'cids':   corpus_ids,
            'cexpr':  json.dumps(cond_expr),
        })
        conn.commit()
        print(f"  Finding saved for {corpus_label}.")
    finally:
        conn.close()


def save_temporal_finding(parent_corpus, base_r2, temporal_r2, n_graphs):
    """Save comparison finding for temporal vs base cross-resolution GNN."""
    delta = temporal_r2 - base_r2 if base_r2 is not None else None
    subclass_list = SUBCLASS_CORPORA_BY_PARENT.get(parent_corpus, G06N_SUBCLASS_CORPORA)
    base_str  = f"{base_r2:.3f}"  if base_r2  is not None else "N/A"
    delta_str = f"{delta:+.3f}"   if delta     is not None else "N/A"

    body = (
        f"Temporal cross-resolution GNN on {parent_corpus} subclass corpora "
        f"({', '.join(subclass_list)}). "
        f"Uses T, T-1, T-2 period triples from same subclass corpus. "
        f"Model: TemporalCrossResGNN (3×SAGEConv shared encoder + graph_ctx fusion). "
        f"Temporal R²={temporal_r2:.3f}. "
        f"Base cross-resolution R²={base_str}. "
        f"Temporal gain: {delta_str}. "
        f"n_graphs={n_graphs}, 80/20 random split on triples."
    )
    confidence = min(0.55 + max(temporal_r2, 0) * 0.35, 0.92)
    cond_expr = {
        'model':        'TemporalCrossResGNN T,T-1,T-2 + graph_ctx',
        'parent_corpus': parent_corpus,
        'temporal_r2':  round(temporal_r2, 4),
        'base_r2':      round(base_r2, 4) if base_r2 is not None else None,
        'temporal_gain': round(delta, 4) if delta is not None else None,
    }

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sci_findings
                (type, title, body, confidence, status, corpus_ids,
                 condition_expression)
            VALUES ('law', %(title)s, %(body)s, %(conf)s, 'active',
                    %(cids)s, %(cexpr)s::jsonb)
        """, {
            'title': f'Temporal Cross-Resolution GNN — {parent_corpus}',
            'body':   body,
            'conf':   confidence,
            'cids':   subclass_list,
            'cexpr':  json.dumps(cond_expr),
        })
        conn.commit()
        print(f"  Temporal finding saved for {parent_corpus}.")
    finally:
        conn.close()


def insert_ml_result(model_name, corpus_id, target, metric_name, metric_value,
                     n_samples=None, notes=None):
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ml_results
                (model_name, corpus_id, target, metric_name, metric_value,
                 n_samples, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (model_name, corpus_id, target, metric_name, float(metric_value),
              n_samples, notes))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='GNN cross-resolution phase transition predictor')
    ap.add_argument(
        '--corpus', default='all',
        help='Corpus ID to train on, or "all" for all subclass corpora combined')
    ap.add_argument(
        '--parent-corpus', default='G06N_quarterly',
        dest='parent_corpus',
        help='Parent corpus to predict (default: G06N_quarterly). '
             'Use H01L_quarterly for H01L subclass run.')
    ap.add_argument(
        '--temporal', action='store_true',
        help='Also run the temporal (T, T-1, T-2) GNN variant')
    args = ap.parse_args()

    parent_corpus = args.parent_corpus.strip()
    corpus_arg    = args.corpus.strip()

    if parent_corpus not in SUBCLASS_CORPORA_BY_PARENT:
        print(f"ERROR: unknown parent corpus '{parent_corpus}'")
        print(f"Valid: {list(SUBCLASS_CORPORA_BY_PARENT.keys())}")
        sys.exit(1)

    subclass_list = SUBCLASS_CORPORA_BY_PARENT[parent_corpus]
    parent_short  = parent_corpus.split('_')[0].lower()  # 'g06n' or 'h01l'

    # Validate --corpus arg against the correct subclass list
    if corpus_arg != 'all' and corpus_arg not in subclass_list:
        print(f"ERROR: '{corpus_arg}' is not a subclass of {parent_corpus}")
        print(f"Valid: all, {', '.join(subclass_list)}")
        sys.exit(1)

    corpus_filter = None if corpus_arg == 'all' else corpus_arg
    corpus_label  = corpus_arg  # 'all' or specific ID

    print(f"\n{'='*60}")
    print(f"arc_gnn.py  parent={parent_corpus}  corpus={corpus_label}"
          f"  temporal={args.temporal}")
    print(f"{'='*60}")

    # ── Base cross-resolution GNN ─────────────────────────────────────────
    print("\nBuilding graphs...")
    graphs = build_graphs(corpus_filter=corpus_filter,
                          parent_corpus=parent_corpus)

    if len(graphs) < 10:
        print(f"WARNING: only {len(graphs)} graphs — results unreliable")

    base_ckpt = str(MODELS_DIR / f'arc_gnn_{parent_short}_cross_resolution.pt')

    print("\nTraining base cross-resolution GNN...")
    best_r2, perm_imp, model, device, test_graphs = \
        train_gnn(graphs, corpus_label=corpus_label, checkpoint=base_ckpt)

    baseline_r2 = compute_tabular_baseline(
        corpus_filter if corpus_filter else None,
        parent_corpus=parent_corpus
    )
    print(f"  Tabular R²: {baseline_r2:.3f}  GNN gain: {best_r2-baseline_r2:+.3f}")

    if best_r2 > 0.2:
        save_finding(corpus_label, best_r2, baseline_r2, perm_imp, len(graphs),
                     parent_corpus=parent_corpus)

    model_name = f'graphsage_cross_resolution_{parent_short}'
    insert_ml_result(model_name, parent_corpus, 'phase_transition_score',
                     'r2', best_r2, n_samples=len(graphs),
                     notes=f'cross-resolution GNN; tabular_r2={baseline_r2:.3f}; '
                           f'gain={best_r2-baseline_r2:+.3f}')
    insert_ml_result(model_name, parent_corpus, 'phase_transition_score',
                     'n_graphs', len(graphs))
    insert_ml_result(model_name, parent_corpus, 'phase_transition_score',
                     'tabular_r2', baseline_r2)

    # ── Temporal cross-resolution GNN (optional) ──────────────────────────
    temporal_r2 = None
    if args.temporal:
        print("\n" + "="*60)
        print(f"Temporal GNN — {parent_corpus}")
        print("="*60)

        temp_ckpt = str(MODELS_DIR / f'arc_gnn_{parent_short}_temporal_cross_resolution.pt')

        temporal_r2 = train_temporal_gnn(
            graphs, corpus_label=corpus_label, checkpoint=temp_ckpt)

        if temporal_r2 is not None:
            temp_model_name = f'graphsage_temporal_cross_resolution_{parent_short}'
            insert_ml_result(temp_model_name, parent_corpus,
                             'phase_transition_score', 'r2', temporal_r2,
                             n_samples=len(graphs),
                             notes=f'temporal (T,T-1,T-2) cross-resolution GNN; '
                                   f'base_r2={best_r2:.3f}; '
                                   f'gain={temporal_r2-best_r2:+.3f}')
            save_temporal_finding(parent_corpus, best_r2, temporal_r2, len(graphs))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — {parent_corpus}")
    print(f"{'='*60}")
    print(f"  Model                              | R²")
    print(f"  {'─'*50}")
    print(f"  Tabular baseline                   | {baseline_r2:.3f}")
    print(f"  Cross-resolution GNN               | {best_r2:.3f}  "
          f"(gain: {best_r2-baseline_r2:+.3f})")
    if temporal_r2 is not None:
        print(f"  Temporal cross-resolution GNN      | {temporal_r2:.3f}  "
              f"(gain vs base: {temporal_r2-best_r2:+.3f})")
    print(f"  {'─'*50}")
    print(f"  n_graphs: {len(graphs)}  |  checkpoint: {base_ckpt}")
    if temporal_r2 is not None:
        print(f"  temporal checkpoint: {temp_ckpt}")
    print(f"{'='*60}\n")
