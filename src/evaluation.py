"""
evaluation.py — LOO evaluation, metrics, statistical tests.
Exact từ notebook Cells 6 (compute_metrics), 7 (LOO), 8 (stats), 10 (GPU LOO).
"""
import time
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.stats import rankdata, wilcoxon as _wilcoxon
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from config import METRIC_KEYS_FULL, RANDOM_SEED


# ── compute_metrics — exact từ notebook Cell 6 ───────────────────────────────

def compute_metrics(scores, test_idx, seed_idx_set,
                    k_values=(5, 10, 20, 50), _n=None):
    """
    Exact từ notebook Cell 6 compute_metrics().

    Protocol:
    - mask: seeds excluded từ candidate pool
    - test_met NOT masked (remains as candidate)
    - rank: rank of test_met among non-seed nodes
    - No leakage: seed_idx_set chỉ chứa seeds, không có test_met

    Args:
        scores       : (N,) float
        test_idx     : G_cc index của test metabolite
        seed_idx_set : set of G_cc indices của seeds
        _n           : hardening — graph size captured at call site
    """
    if _n is None: _n = len(scores)

    mask = np.ones(_n, dtype=bool)
    for si in seed_idx_set: mask[si] = False

    labels = np.zeros(_n)
    labels[test_idx] = 1.0

    sc = scores[mask]
    lb = labels[mask]

    if sc.sum() < 1e-12 or lb.sum() < 1: return None
    try:
        auc = roc_auc_score(lb, sc)
    except ValueError:
        return None

    mi = np.where(mask)[0]
    tp = np.where(mi == test_idx)[0]
    if len(tp) == 0: return None

    rank = float(rankdata(-sc, method='average')[tp[0]])
    out  = {'auc': float(auc), 'rank': rank, 'mrr': 1.0 / rank}
    for k in k_values: out[f'r@{k}'] = 1.0 if rank <= k else 0.0
    return out


# ── LOO evaluation (CPU) — exact từ notebook Cell 7 ──────────────────────────

def run_loo_eval(disease_set, method_fn, node_idx, N, label=''):
    """
    Exact từ notebook Cell 7 run_loo_eval().
    method_fn(seed_nodes) → scores (N,)
    """
    rows = []
    for disease, mets in tqdm(disease_set.items(), desc=label or 'LOO'):
        valid = [m for m in mets if m in node_idx]
        if len(valid) < 3: continue
        loo_res = []
        for i, test_met in enumerate(valid):
            seeds        = [m for j, m in enumerate(valid) if j != i]
            # EXACT: seed_idx_set = {node_idx[s] for s in seeds}
            # (không có test_met — no leakage)
            seed_idx_set = {node_idx[s] for s in seeds}
            test_idx     = node_idx[test_met]
            try:
                scores = method_fn(seeds)
                m = compute_metrics(scores, test_idx, seed_idx_set, _n=N)
                if m: loo_res.append(m)
            except Exception:
                pass
        if loo_res:
            row = {k: float(np.mean([r[k] for r in loo_res]))
                   for k in METRIC_KEYS_FULL}
            row['disease'] = disease
            row['n_mets']  = len(mets)   # len(mets) per notebook Cell 7
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


# ── Wilcoxon — exact từ notebook Cell 8 wilcoxon_table() ────────────────────

def wilcoxon_table(df_a, df_b, label,
                   metrics=None, method_a='CTQW-PRO', method_b='PROFANCY'):
    """
    Exact từ notebook Cell 8.
    Bonferroni: p_bonf = min(p * len(metrics), 1.0)
    """
    if metrics is None:
        metrics = ['auc','mrr','r@5','r@10','r@20','r@50']

    shared = sorted(set(df_a['disease']) & set(df_b['disease']))
    if len(shared) < 5:
        print(f'  {label}: {len(shared)} shared diseases — skip')
        return None

    da = df_a.set_index('disease')
    db = df_b.set_index('disease')
    rows = []
    print(f'\n--- {label} (n={len(shared)}) ---')
    print(f"  {'Metric':<8} {method_b:>16} {method_a:>16} "
          f"{'Delta':>8} {'p_bonf':>10}  Sig")

    for met in metrics:
        a = da.loc[shared, met].values   # method_a (CTQW-PRO)
        b = db.loc[shared, met].values   # method_b (PROFANCY)
        delta = float((a - b).mean())
        try:
            _, p = _scipy_stats.wilcoxon(
                a, b, alternative='two-sided', zero_method='wilcox')
        except Exception:
            p = np.nan
        p_bonf = (min(float(p) * len(metrics), 1.0)
                  if not np.isnan(p) else np.nan)

        def _sig(v):
            if np.isnan(v): return ''
            return ('***' if v<0.001 else '**' if v<0.01
                    else '*' if v<0.05 else '.' if v<0.1 else 'ns')

        print(f'  {met:<8} {b.mean():>8.4f}+/-{b.std():>6.4f} '
              f'{a.mean():>8.4f}+/-{a.std():>6.4f} '
              f'{delta:>+8.4f} {p_bonf:>10.4g}  {_sig(p_bonf)}')
        rows.append({
            'source': label, 'metric': met, 'n': len(shared),
            f'{method_b}_mean': float(b.mean()),
            f'{method_a}_mean': float(a.mean()),
            'delta': delta,
            'p_bonf': float(p_bonf) if not np.isnan(p_bonf) else np.nan,
        })
    return pd.DataFrame(rows)


# ── Bootstrap CI — exact từ notebook Cell 8 ──────────────────────────────────

def bootstrap_ci(df, metric='mrr', n_bootstrap=1000, ci=95.0):
    """Exact từ notebook Cell 8."""
    rng  = np.random.default_rng(RANDOM_SEED)
    vals = df[metric].values
    mean = float(vals.mean())
    boot = np.array([
        rng.choice(vals, len(vals), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return mean, lo, hi


# ── Win counts — exact từ notebook Cell 8 ────────────────────────────────────

def win_counts(df_a, df_b, metric='auc', name_a='CTQW-PRO', name_b='PROFANCY'):
    """Exact từ notebook Cell 8."""
    shared = sorted(set(df_a['disease']) & set(df_b['disease']))
    da = df_a.set_index('disease')
    db = df_b.set_index('disease')
    wins = {name_a: 0, name_b: 0, 'tie': 0}
    for d in shared:
        diff = da.loc[d, metric] - db.loc[d, metric]
        if abs(diff) < 1e-9: wins['tie'] += 1
        elif diff > 0:       wins[name_a] += 1
        else:                wins[name_b] += 1
    wins['n_shared'] = len(shared)
    return wins


# ── Print results table — exact từ notebook Cell 10 print_results() ──────────

def print_results_table(results_dict, label, method_order=None):
    """
    Fix: explicit None check (không dùng `or` trên DataFrame).
    Format exact từ notebook Cell 10 print_results().
    """
    order = method_order or list(results_dict.keys())

    # Explicit None check
    base = None
    for nm in order:
        candidate = results_dict.get(nm)
        if candidate is not None and not candidate.empty:
            base = candidate
            break

    bm = base['mrr'].mean()  if base is not None else 0
    br = base['r@20'].mean() if base is not None else 0
    n  = len(base) if base is not None else 0

    print(f'\n=== {label} (n={n}) ===')
    print(f"{'Method':<22} {'AUC':>8} {'MRR':>8} "
          f"{'R@5':>7} {'R@10':>7} {'R@20':>7}")
    print('-' * 57)

    for nm in order:
        df = results_dict.get(nm)
        if df is None or df.empty: continue
        dm   = df['mrr'].mean() - bm
        dr   = df['r@20'].mean() - br
        mark = ' ◄' if (dm > 0.002 or dr > 0.003) and nm != order[0] else ''
        print(f"{nm:<22} {df['auc'].mean():>8.4f} {df['mrr'].mean():>8.4f} "
              f"{df['r@5'].mean():>7.4f} {df['r@10'].mean():>7.4f} "
              f"{df['r@20'].mean():>7.4f}{mark}")

