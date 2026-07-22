"""
05_supplementary.py — Supplementary experiments và figures cho paper CTQW-PRO.

Tạo ra:
  Supplementary Fig. S1 — γ grid search (NH-CTQW-PRO)
  Supplementary Fig. S2 — Driven CTQW-PRO t-sweep
  Supplementary Fig. S3 — Driven n_steps × α heatmap
  Supplementary Table S0 — Danh sách 52 cofactor species
  Supplementary Table S1 — Ablation results (đã tạo bởi 02_ablation_graph.py)

Usage:
  python 05_supplementary.py             # chạy tất cả (mặc định)
  python 05_supplementary.py --s1        # chỉ γ grid
  python 05_supplementary.py --s2        # chỉ t-sweep + Driven t-sweep
  python 05_supplementary.py --s3        # chỉ Driven n_steps×alpha
  python 05_supplementary.py --tables    # chỉ Table S0

Notes:
  - S1 và S2 (t-sweep) đọc từ CSV đã lưu bởi grid_gamma_cell.py và grid_t_cell.py
  - S3 đọc từ CSV đã lưu bởi grid_driven_cell.py / grid_driven_ultrafast_cell.py
  - Nếu CSV chưa tồn tại, script tự chạy grid search trước khi plot
"""
import sys, os, argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────
for _p in [str(Path(__file__).resolve().parent.parent / 'src'),
           '/content/project/src',
           '/content/drive/MyDrive/CTQW for metabolites/src']:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p); break

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import time
import warnings; warnings.filterwarnings('ignore')

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED, NH_GAMMA,
                    RECON3D_COFACTORS, DRIVEN_N_STEPS, DRIVEN_ALPHA)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR = RESULTS_DIR / 'figures'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style — nhất quán với 04_figures.py ──────────────────────────
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'legend.frameon': False,
    'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,  'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 1.2,   'patch.linewidth': 0.4,
    'pdf.fonttype': 42,       'ps.fonttype': 42,
    'figure.dpi': 150,        'savefig.dpi': 300,
    'savefig.bbox': 'tight',  'savefig.pad_inches': 0.05,
})

C = {
    'CTQW':    '#1A7DC4',
    'NH':      '#8B4AAD',
    'Driven':  '#28996B',
    'HMDB':    '#E63946',
    'SMPDB':   '#2D6A4F',
    'gray':    '#888888',
    'grid':    '#CCCCCC',
}
W1 = 3.46; W15 = 5.2; W2 = 7.09

def savefig(fig, name):
    for ext in ('pdf', 'png'):
        p = FIGS_DIR / f'{name}.{ext}'
        fig.savefig(p, dpi=300)
        print(f'  Saved: {p}')


# ════════════════════════════════════════════════════════════════
# SETUP — dùng chung cho S1, S2, S3 nếu cần chạy grid
# ════════════════════════════════════════════════════════════════
def setup_pipeline():
    from graph import (parse_recon3d, build_gcc, build_gpro,
                       build_hmdb_to_recon_initial, augment_hmdb_to_recon,
                       compute_eigendecomp)
    from eval_sets import (parse_hmdb, build_hmdb_lookups,
                           build_cofactors_set, build_eval_set1, build_eval_set3)

    print('Setup pipeline...')
    recon_data   = parse_recon3d()
    G_cc, graph_nodes, N, node_idx, A_cc, degrees = build_gcc(recon_data)
    met_info     = recon_data['met_info']
    pathway_mets = recon_data['pathway_mets']

    (G_pro, pro_nodes, N_PRO, idx_pro,
     A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

    eigvals, eigvecs = compute_eigendecomp(
        A_pro, CACHE_DIR / 'gpro_eigdecomp.npz')

    hmdb_data        = parse_hmdb()
    hmdb_metabolites = hmdb_data['metabolites']
    hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)
    hmdb_to_recon    = build_hmdb_to_recon_initial(met_info, node_idx)
    augment_hmdb_to_recon(
        hmdb_to_recon, met_info, node_idx,
        hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
        hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'])
    COFACTORS = build_cofactors_set(hmdb_metabolites)
    eval_set1, _ = build_eval_set1(
        hmdb_metabolites, hmdb_lookups, hmdb_to_recon, node_idx, COFACTORS)
    eval_set3     = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx)
    print(f'  HMDB+CTD: {len(eval_set1)}, SMPDB: {len(eval_set3)}')

    return dict(
        N=N, node_idx=node_idx, N_PRO=N_PRO, idx_pro=idx_pro,
        A_pro=A_pro, deg_pro=deg_pro, pro_nodes=pro_nodes,
        eigvals=eigvals, eigvecs=eigvecs,
        _pro_src=_pro_src, _pro_dst=_pro_dst,
        eval_set1=eval_set1, eval_set3=eval_set3,
        COFACTORS=COFACTORS,
    )


# ════════════════════════════════════════════════════════════════
# S1 — γ GRID SEARCH (NH-CTQW-PRO)
# ════════════════════════════════════════════════════════════════
GAMMA_GRID_S1 = [10, 15, 22, 30, 40, 50, 60, 80, 150, 349]

def run_s1_experiment(ctx):
    """Chạy γ grid search nếu CSV chưa có."""
    from methods import make_nh_pro, make_ctqw_pro
    from evaluation import run_loo_eval

    N, node_idx   = ctx['N'], ctx['node_idx']
    N_PRO         = ctx['N_PRO']
    idx_pro       = ctx['idx_pro']
    A_pro         = ctx['A_pro']
    pro_nodes     = ctx['pro_nodes']
    eigvals       = ctx['eigvals']
    eigvecs       = ctx['eigvecs']
    _pro_src      = ctx['_pro_src']
    _pro_dst      = ctx['_pro_dst']
    eval_set1     = ctx['eval_set1']
    eval_set3     = ctx['eval_set3']

    # CTQW baseline
    run_ctqw = make_ctqw_pro(
        eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)
    ctqw_fn = lambda seeds: run_ctqw(seeds, [T_FIXED])[T_FIXED]

    base_rows = []
    for label, dset in [('HMDB+CTD', eval_set1), ('SMPDB', eval_set3)]:
        df = run_loo_eval(dset, ctqw_fn, node_idx, N, label=f'CTQW/{label}')
        if df is not None:
            tmp = df.copy(); tmp['gamma'] = 'baseline'; tmp['dataset'] = label
            base_rows.append(tmp)
    if base_rows:
        pd.concat(base_rows).to_csv(
            RESULTS_DIR / 'grid_gamma_baseline.csv', index=False)

    # γ grid
    all_rows = []
    for gamma in GAMMA_GRID_S1:
        t0 = time.time()
        print(f'  γ={gamma}...', end=' ', flush=True)
        run_nh = make_nh_pro(
            A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
            RECON3D_COFACTORS, pro_nodes, float(gamma), T_FIXED)
        print(f'{time.time()-t0:.1f}s')
        for label, dset in [('HMDB+CTD', eval_set1), ('SMPDB', eval_set3)]:
            df = run_loo_eval(dset, run_nh, node_idx, N,
                              label=f'NH(γ={gamma})/{label}')
            if df is not None and not df.empty:
                tmp = df.copy(); tmp['gamma'] = gamma; tmp['dataset'] = label
                all_rows.append(tmp)

    if all_rows:
        pd.concat(all_rows).to_csv(
            RESULTS_DIR / 'grid_gamma_search.csv', index=False)


def plot_s1():
    """Supplementary Fig. S1 — γ grid search."""
    print('\n── Supplementary Fig. S1: γ grid search ──')

    csv_path = RESULTS_DIR / 'grid_gamma_search.csv'
    base_path = RESULTS_DIR / 'grid_gamma_baseline.csv'

    if not csv_path.exists():
        print('  CSV not found — running experiment first...')
        ctx = setup_pipeline()
        run_s1_experiment(ctx)

    df    = pd.read_csv(csv_path)
    # Convert gamma column: might be stored as float
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce')
    df = df.dropna(subset=['gamma'])

    df_base = pd.read_csv(base_path) if base_path.exists() else None

    gammas = sorted(df['gamma'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(W15, 2.2),
                             gridspec_kw={'wspace': 0.38})

    for ax, (label, col) in zip(axes, [('HMDB+CTD', C['HMDB']),
                                        ('SMPDB',    C['SMPDB'])]):
        sub = df[df['dataset'] == label]
        mrrs = [sub[sub['gamma'] == g]['mrr'].mean() for g in gammas]
        mrrs = [m if not np.isnan(m) else 0 for m in mrrs]

        # Baseline
        base_mrr = None
        if df_base is not None:
            b = df_base[df_base['dataset'] == label]
            if not b.empty:
                base_mrr = float(b['mrr'].mean())

        ax.plot(gammas, mrrs, 'o-', color=col, lw=1.4,
                markersize=4, markerfacecolor='white',
                markeredgewidth=1.2, zorder=4, label='NH-CTQW-PRO')

        if base_mrr is not None:
            ax.axhline(base_mrr, ls='--', lw=1.0, color=C['CTQW'],
                       label='CTQW-PRO', zorder=3)

        # Mark γ=22 (paper choice)
        if NH_GAMMA in gammas:
            idx22 = gammas.index(NH_GAMMA)
            ax.scatter([NH_GAMMA], [mrrs[idx22]], marker='*', s=80,
                       color='#FFD700', edgecolors='#888', linewidths=0.6,
                       zorder=6, label=f'γ=k̄={NH_GAMMA} (paper)')

        ax.set_xlabel('γ (decay rate)', labelpad=3)
        ax.set_ylabel('MRR', labelpad=3)
        ax.set_title(label, fontweight='bold')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_xticks(gammas)
        ax.set_xticklabels([str(int(g)) for g in gammas],
                           rotation=45, ha='right', fontsize=6)
        ax.grid(axis='y', linewidth=0.35, alpha=0.5, linestyle=':', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=6)

    fig.suptitle('Supplementary Fig. S1 — NH-CTQW-PRO γ grid search\n'
                 '★ = γ = k̄ = 22 (paper default)', fontsize=7.5, y=1.04)
    plt.tight_layout()
    savefig(fig, 'figS1_gamma_grid')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# S2 — t GRID SEARCH + DRIVEN t-SWEEP
# ════════════════════════════════════════════════════════════════
T_GRID_S2    = [0.05, 0.1, 0.2, 0.5]
T_SWEEP_S2   = [0.1, 0.5, 1.0, 2.0]   # Driven t-sweep từ session thực nghiệm

def plot_s2():
    """Supplementary Fig. S2 — t grid (CTQW) + Driven t-sweep."""
    print('\n── Supplementary Fig. S2: t grid + Driven t-sweep ──')

    csv_path = RESULTS_DIR / 'grid_t_search.csv'
    if not csv_path.exists():
        print('  WARNING: grid_t_search.csv not found.')
        print('  Chạy grid_t_cell.py trước để tạo CSV.')
        return

    df_t  = pd.read_csv(csv_path)
    # Driven t-sweep results (từ session thực nghiệm của bạn)
    # CTQW:  t=0.1→0.2043, t=0.5→0.1596, t=1.0→0.0691, t=2.0→0.0645
    # Driven: t=0.1→0.2568, t=0.5→0.0967, t=1.0→0.0563, t=2.0→0.0645
    sweep_ctqw   = {0.1: 0.2043, 0.5: 0.1596, 1.0: 0.0691, 2.0: 0.0645}
    sweep_driven = {0.1: 0.2568, 0.5: 0.0967, 1.0: 0.0563, 2.0: 0.0645}

    fig, axes = plt.subplots(1, 2, figsize=(W15, 2.2),
                             gridspec_kw={'wspace': 0.40})

    # Left: t grid (CTQW-PRO, HMDB+CTD)
    ax = axes[0]
    t_vals = sorted(df_t['t'].unique())
    mrrs   = [df_t[df_t['t'] == t]['mrr'].mean() for t in t_vals]

    ax.plot(t_vals, mrrs, 'o-', color=C['CTQW'], lw=1.4,
            markersize=5, markerfacecolor='white',
            markeredgewidth=1.2, zorder=4)

    # Mark best (t=0.1)
    best_idx = int(np.argmax(mrrs))
    ax.scatter([t_vals[best_idx]], [mrrs[best_idx]], marker='*', s=100,
               color='#FFD700', edgecolors='#888', linewidths=0.6,
               zorder=6, label=f't*={t_vals[best_idx]} (selected)')

    ax.set_xlabel('Evolution time t', labelpad=3)
    ax.set_ylabel('MRR', labelpad=3)
    ax.set_title('CTQW-PRO t grid\n(HMDB+CTD)', fontweight='bold')
    ax.set_xticks(t_vals)
    ax.set_xticklabels([str(t) for t in t_vals])
    ax.grid(axis='y', linewidth=0.35, alpha=0.5, linestyle=':', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6)

    # Right: Driven t-sweep (SMPDB, 30-disease sample)
    ax = axes[1]
    ts_plot = sorted(sweep_ctqw.keys())

    ax.plot(ts_plot, [sweep_ctqw[t] for t in ts_plot],
            'o-', color=C['CTQW'], lw=1.4, markersize=4,
            markerfacecolor='white', markeredgewidth=1.2,
            zorder=4, label='CTQW-PRO')
    ax.plot(ts_plot, [sweep_driven[t] for t in ts_plot],
            's--', color=C['Driven'], lw=1.4, markersize=4,
            markerfacecolor='white', markeredgewidth=1.2,
            zorder=4, label='Driven CTQW-PRO')

    # Mark t=0.1
    ax.axvline(0.1, ls=':', lw=0.8, color=C['gray'], zorder=3)
    ymin_driven = min(sweep_ctqw.values())
    ax.text(0.12, ymin_driven * 1.05, 't=0.1',
            fontsize=6, color=C['gray'], va='bottom')

    ax.set_xlabel('Evolution time t (per stride)', labelpad=3)
    ax.set_ylabel('MRR', labelpad=3)
    ax.set_title('Driven CTQW-PRO t-sweep\n(SMPDB, n=30 sample)', fontweight='bold')
    ax.set_xticks(ts_plot)
    ax.grid(axis='y', linewidth=0.35, alpha=0.5, linestyle=':', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6)

    fig.suptitle('Supplementary Fig. S2 — Evolution time sensitivity',
                 fontsize=7.5, y=1.04)
    plt.tight_layout()
    savefig(fig, 'figS2_t_search')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# S3 — DRIVEN n_steps × alpha HEATMAP
# ════════════════════════════════════════════════════════════════
N_STEPS_GRID_S3 = [1, 2, 3, 5, 10, 15, 20]
ALPHA_GRID_S3   = [0.1, 0.2, 0.3, 0.5, 0.8]

def run_s3_experiment(ctx):
    """Chạy Driven grid nếu CSV chưa có."""
    from evaluation import compute_metrics

    N, node_idx   = ctx['N'], ctx['node_idx']
    N_PRO         = ctx['N_PRO']
    idx_pro       = ctx['idx_pro']
    eigvals       = ctx['eigvals']
    eigvecs       = ctx['eigvecs']
    _pro_src      = ctx['_pro_src']
    _pro_dst      = ctx['_pro_dst']
    eval_set3     = ctx['eval_set3']

    phases = np.exp(-1j * eigvals * T_FIXED)
    N_DISEASES = 50
    all_d  = sorted(eval_set3.items(), key=lambda x: -len(x[1]))
    step   = max(1, len(all_d) // N_DISEASES)
    sample = dict(list(all_d)[i] for i in range(0, len(all_d), step))
    sample = dict(list(sample.items())[:N_DISEASES])
    print(f'  {len(sample)} diseases sample')

    def run_driven(seeds, n_steps, alpha):
        valid = [idx_pro[s] for s in seeds if s in idx_pro]
        if not valid: return np.zeros(N)
        ps = np.zeros(N_PRO, dtype=complex)
        ps[valid] = 1.0 / np.sqrt(len(valid))
        psi = ps.copy()
        for _ in range(n_steps):
            walked = eigvecs @ (phases * (eigvecs.conj().T @ psi))
            psi    = (1-alpha)*walked + alpha*ps
            nrm    = np.linalg.norm(psi)
            if nrm > 1e-9: psi /= nrm
        sc = np.zeros(N)
        sc[_pro_dst] = (np.abs(psi)**2)[_pro_src]
        return sc

    save_rows = []
    total = len(N_STEPS_GRID_S3) * len(ALPHA_GRID_S3)
    done  = 0
    for n_steps in N_STEPS_GRID_S3:
        for alpha in ALPHA_GRID_S3:
            rows = []
            for disease, mets in sample.items():
                valid = [m for m in mets if m in node_idx]
                if len(valid) < 3: continue
                fold_res = []
                for i, test_met in enumerate(valid):
                    seeds    = [m for j, m in enumerate(valid) if j != i]
                    seed_set = {node_idx[s] for s in seeds}
                    try:
                        sc = run_driven(seeds, n_steps, alpha)
                        m  = compute_metrics(sc, node_idx[test_met],
                                             seed_set, _n=N)
                        if m: fold_res.append(m)
                    except Exception:
                        pass
                if fold_res:
                    row = {k: float(np.mean([r[k] for r in fold_res]))
                           for k in ['auc','mrr','r@5','r@20']}
                    row['disease'] = disease
                    rows.append(row)
            done += 1
            df = pd.DataFrame(rows) if rows else None
            if df is not None and not df.empty:
                tmp = df.copy()
                tmp['n_steps'] = n_steps; tmp['alpha'] = alpha
                save_rows.append(tmp)
            print(f'  [{done}/{total}] n_steps={n_steps}, α={alpha}')

    if save_rows:
        pd.concat(save_rows).to_csv(
            RESULTS_DIR / 'grid_driven_search.csv', index=False)


def plot_s3():
    """Supplementary Fig. S3 — Driven n_steps × alpha heatmap."""
    print('\n── Supplementary Fig. S3: Driven n_steps × alpha ──')

    csv_path  = RESULTS_DIR / 'grid_driven_search.csv'
    base_path = RESULTS_DIR / 'grid_driven_baseline.csv'

    if not csv_path.exists():
        print('  CSV not found — running experiment first...')
        ctx = setup_pipeline()
        run_s3_experiment(ctx)

    df = pd.read_csv(csv_path)
    df_base = pd.read_csv(base_path) if base_path.exists() else None
    base_mrr = float(df_base['mrr'].mean()) if df_base is not None else 0

    # Build MRR matrix (n_steps × alpha)
    n_steps_vals = sorted(df['n_steps'].unique())
    alpha_vals   = sorted(df['alpha'].unique())
    mat = np.full((len(n_steps_vals), len(alpha_vals)), np.nan)
    for i, ns in enumerate(n_steps_vals):
        for j, a in enumerate(alpha_vals):
            sub = df[(df['n_steps'] == ns) & (np.isclose(df['alpha'], a))]
            if not sub.empty:
                mat[i, j] = float(sub['mrr'].mean())

    # Delta over baseline
    delta_mat = mat - base_mrr

    fig, axes = plt.subplots(1, 2, figsize=(W15, 2.6),
                             gridspec_kw={'wspace': 0.45})

    for ax, (data, title, fmt, cmap) in zip(axes, [
        (mat,       'MRR',       '.3f', 'Blues'),
        (delta_mat, 'ΔMRR vs CTQW-PRO', '+.3f', 'RdYlGn'),
    ]):
        vmax = np.nanmax(np.abs(data)) if 'Δ' in title else np.nanmax(data)
        vmin = -vmax if 'Δ' in title else np.nanmin(data)
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        # Cell text
        best_val = np.nanmax(data) if 'Δ' in title else np.nanmax(data)
        for i in range(len(n_steps_vals)):
            for j in range(len(alpha_vals)):
                v = data[i, j]
                if np.isnan(v): continue
                bg_norm = (v - vmin) / (vmax - vmin + 1e-9)
                txt_c   = 'white' if bg_norm > 0.6 else '#222'
                is_best = np.isclose(v, best_val, atol=0.001)
                weight  = 'bold' if is_best else 'normal'
                ax.text(j, i, f'{v:{fmt}}',
                        ha='center', va='center',
                        fontsize=6.5, color=txt_c, fontweight=weight)

        # Mark paper params (n_steps=2, alpha=0.5)
        if DRIVEN_N_STEPS in n_steps_vals and DRIVEN_ALPHA in alpha_vals:
            ri = n_steps_vals.index(DRIVEN_N_STEPS)
            ci = [round(a, 2) for a in alpha_vals].index(round(DRIVEN_ALPHA, 2))
            rect = plt.Rectangle(
                (ci - 0.5, ri - 0.5), 1, 1,
                fill=False, edgecolor='#FFD700',
                linewidth=2.0, zorder=5)
            ax.add_patch(rect)

        ax.set_xticks(range(len(alpha_vals)))
        ax.set_xticklabels([f'α={a}' for a in alpha_vals], fontsize=6.5)
        ax.set_yticks(range(len(n_steps_vals)))
        ax.set_yticklabels([f'{n}' for n in n_steps_vals], fontsize=6.5)
        ax.set_xlabel('α (re-injection weight)', labelpad=3)
        ax.set_ylabel('n_steps', labelpad=3)
        ax.set_title(title, fontweight='bold')

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5.5)

    # Legend for gold box
    gold_patch = mpatches.Patch(
        facecolor='none', edgecolor='#FFD700', linewidth=2,
        label=f'Paper: n_steps={DRIVEN_N_STEPS}, α={DRIVEN_ALPHA}')
    fig.legend(handles=[gold_patch], loc='lower center',
               fontsize=6.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        'Supplementary Fig. S3 — Driven CTQW-PRO stability analysis\n'
        f'SMPDB (n=50 sample). ★ = best. ■ = paper params.',
        fontsize=7.5, y=1.04)
    plt.tight_layout()
    savefig(fig, 'figS3_driven_grid')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# TABLE S0 — Cofactor species list
# ════════════════════════════════════════════════════════════════
def make_table_s0():
    """Supplementary Table S0 — 52 cofactor species."""
    print('\n── Supplementary Table S0: Cofactor list ──')

    # Categorize theo nhóm hóa học
    categories = {
        'Energy carriers (nucleotides)': [
            'atp','adp','amp','gtp','gdp','gmp',
            'ctp','cdp','cmp','utp','udp','ump',
            'datp','dadp','damp',
        ],
        'Electron carriers': [
            'nad','nadh','nadp','nadph',
            'fad','fadh2','fmn','fmnh2',
        ],
        'Acyl carriers': [
            'coa','accoa','q','qh2',
        ],
        'Inorganic ions': [
            'na1','k1','cl','ca2','mg2',
            'fe2','fe3','zn2','cu2','mn2',
        ],
        'Small molecules': [
            'h2o','h','co2','o2','pi','ppi','hco3',
            'h2o2','nh3','nh4','so4','no','h2','oh1','h2s',
        ],
    }

    rows = []
    for cat, species in categories.items():
        for sp in species:
            rows.append({'Category': cat, 'Recon3D ID': sp})

    df_s0 = pd.DataFrame(rows)

    # Save as CSV
    out_csv = RESULTS_DIR / 'tableS0_cofactors.csv'
    df_s0.to_csv(out_csv, index=False)
    print(f'  Saved: {out_csv}')
    print(f'  Total: {len(df_s0)} species in {len(categories)} categories')

    # Verify count
    all_sp = [sp for lst in categories.values() for sp in lst]
    missing = set(RECON3D_COFACTORS) - set(all_sp)
    extra   = set(all_sp) - set(RECON3D_COFACTORS)
    if missing: print(f'  WARNING: {len(missing)} species in config but not table: {missing}')
    if extra:   print(f'  WARNING: {len(extra)} species in table but not config: {extra}')
    if not missing and not extra:
        print(f'  ✅ Table matches RECON3D_COFACTORS exactly ({len(all_sp)} species)')

    return df_s0


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Generate supplementary figures and tables')
    parser.add_argument('--s1',     action='store_true', help='γ grid (Fig S1)')
    parser.add_argument('--s2',     action='store_true', help='t grid (Fig S2)')
    parser.add_argument('--s3',     action='store_true', help='Driven grid (Fig S3)')
    parser.add_argument('--tables', action='store_true', help='Table S0')
    parser.add_argument('--all',    action='store_true', help='All (default)')

    # Handle Jupyter/Colab context where sys.argv contains kernel args
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    run_all = args.all or not any([args.s1, args.s2, args.s3, args.tables])

    if run_all or args.tables:
        make_table_s0()

    if run_all or args.s1:
        plot_s1()

    if run_all or args.s2:
        plot_s2()

    if run_all or args.s3:
        plot_s3()

    print(f'\n✅ Done. Figures saved to: {FIGS_DIR}')


if __name__ == '__main__':
    main()
