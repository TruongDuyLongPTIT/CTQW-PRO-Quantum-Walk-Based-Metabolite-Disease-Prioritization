"""
01_main_results.py — Main experiment.
Table 1: CTQW vs RWR (G_cc)    [chạy trước]
Table 2: CTQW-PRO vs PROFANCY (G_pro)
Table 3: Driven CTQW-PRO vs CTQW-PRO

Closely follows notebook Cell 7 (LOO) and Cell 8 (stats).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    RESULTS_DIR, CACHE_DIR,
    T_FIXED, RWR_ALPHA, DRIVEN_N_STEPS, DRIVEN_ALPHA,
    NH_GAMMA, RRF_K,
    RECON3D_COFACTORS, COFACTORS_FALLBACK,
    MIN_METS, METRIC_KEYS_FULL,
    SMPDB_MET_DIR, SMPDB_PW_DIR,
    PATH_SMPDB_PW, PATH_SMPDB_MET,
)
from graph import (
    parse_recon3d, build_gcc, build_gpro,
    build_hmdb_to_recon_initial, augment_hmdb_to_recon,
    compute_eigendecomp,
)
from eval_sets import (
    parse_hmdb, build_hmdb_lookups, build_cofactors_set,
    build_eval_set1, build_eval_set2, build_eval_set3,
)
from methods import (
    run_rwr, make_profancy, make_ctqw_pro, make_ctqw_gcc,
    make_nh_pro, make_rrf,
    build_gpu_methods, build_psi_batch,
)
from evaluation import (
    run_loo_eval, run_driven_eval,
    wilcoxon_table, bootstrap_ci, win_counts,
    print_results_table,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Build graph (exact từ notebook Cell 2)
# ═══════════════════════════════════════════════════════════════
print('='*60); print('STEP 1 — Build graph')

recon_data   = parse_recon3d()
G_cc, graph_nodes, N, node_idx, A_cc, degrees = build_gcc(recon_data)
met_info     = recon_data['met_info']
pathway_mets = recon_data['pathway_mets']

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

# Transition matrix G_pro — exact từ notebook Cell 6
deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro        = A_pro / deg_pro_safe[:, None]

# Transition matrix G_cc — for RWR baseline
deg_cc_safe  = np.where(A_cc.sum(1) > 0, A_cc.sum(1), 1.0)
P_cc         = A_cc / deg_cc_safe[:, None]

print(f'  Graph: {N} nodes, {G_cc.number_of_edges()} edges')
print(f'  G_pro: {N_PRO} nodes ({sum(1 for nd in pro_nodes if nd.startswith("__PATH__"))} pathway), '
      f'{G_pro.number_of_edges()} edges')

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Build eval sets (exact từ notebook Cells 3,4,5)
# ═══════════════════════════════════════════════════════════════
print('\nSTEP 2 — Build eval sets')

hmdb_data        = parse_hmdb()
hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)

# hmdb_to_recon: initial mapping then augment (exact từ notebook Cell 2 + Cell 3)
hmdb_to_recon = build_hmdb_to_recon_initial(met_info, node_idx)
n_ik, n_nm = augment_hmdb_to_recon(
    hmdb_to_recon, met_info, node_idx,
    hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
    hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'],
)
print(f'  hmdb_to_recon: +{n_ik} IK, +{n_nm} name → {len(hmdb_to_recon)} total')

# COFACTORS name-based set (exact từ notebook Cell 3)
COFACTORS = build_cofactors_set(hmdb_metabolites)

# Build eval sets
eval_set1, disease_canonical = build_eval_set1(
    hmdb_metabolites, hmdb_lookups, hmdb_to_recon, node_idx, COFACTORS)
eval_set2 = build_eval_set2(
    hmdb_metabolites, hmdb_lookups, hmdb_to_recon,
    node_idx, COFACTORS, disease_canonical)
eval_set3 = build_eval_set3(
    hmdb_metabolites, hmdb_to_recon, node_idx)

print(f'  eval_set1 (HMDB+CTD): {len(eval_set1)} diseases')
print(f'  eval_set2 (MarkerDB): {len(eval_set2)} diseases')
print(f'  eval_set3 (SMPDB):    {len(eval_set3)} diseases')

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Eigendecomposition (exact từ notebook Cell 6)
# ═══════════════════════════════════════════════════════════════
print('\nSTEP 3 — Eigendecomposition')
Apro_eigvals, Apro_eigvecs = compute_eigendecomp(
    A_pro, CACHE_DIR / 'gpro_eigdecomp.npz')
Acc_eigvals, Acc_eigvecs   = compute_eigendecomp(
    A_cc, CACHE_DIR / 'gcc_eigdecomp.npz')
print('  Done.')

# ═══════════════════════════════════════════════════════════════
# STEP 4 — GPU setup
# ═══════════════════════════════════════════════════════════════
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    gpu_ok = True
except ImportError:
    gpu_ok = False

# ═══════════════════════════════════════════════════════════════
# TABLE 1 — RWR vs CTQW on G_cc  [FIRST: simpler, establishes baseline]
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('[Table 1] RWR vs CTQW on G_cc...')

# G_cc methods — closures capturing N at definition time
_rwr_fn   = lambda seeds: run_rwr(seeds, P_cc, node_idx, N)
_ctqw_gcc = make_ctqw_gcc(Acc_eigvals, Acc_eigvecs, N)
_ctqw_gcc_fn = lambda seeds: _ctqw_gcc(seeds, node_idx)

all_t1 = {}
for label, dset in [('HMDB+CTD', eval_set1),
                     ('MarkerDB', eval_set2),
                     ('SMPDB',    eval_set3)]:
    t0 = time.time()
    df_rwr  = run_loo_eval(dset, _rwr_fn,   node_idx, N, label=f'RWR/{label}')
    df_ctqw = run_loo_eval(dset, _ctqw_gcc_fn, node_idx, N, label=f'CTQW/{label}')
    all_t1[label] = {'RWR': df_rwr, 'CTQW': df_ctqw}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

# ═══════════════════════════════════════════════════════════════
# TABLE 2 — PROFANCY vs CTQW-PRO on G_pro
# ═══════════════════════════════════════════════════════════════
print('\n[Table 2] PROFANCY vs CTQW-PRO on G_pro...')

# PROFANCY: closure captures all state (hardened _n=N)
run_profancy = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
# CTQW-PRO: closure
run_ctqw_pro = make_ctqw_pro(
    Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)

_ctqw_pro_fn = lambda seeds: run_ctqw_pro(seeds, [T_FIXED])[T_FIXED]

all_t2 = {}
for label, dset in [('HMDB+CTD', eval_set1),
                     ('MarkerDB', eval_set2),
                     ('SMPDB',    eval_set3)]:
    t0 = time.time()
    df_prof = run_loo_eval(dset, run_profancy, node_idx, N,
                           label=f'PROFANCY/{label}')
    df_ctqw = run_loo_eval(dset, _ctqw_pro_fn, node_idx, N,
                           label=f'CTQW-PRO/{label}')
    all_t2[label] = {'PROFANCY': df_prof, 't=0.1': df_ctqw}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

# ═══════════════════════════════════════════════════════════════
# TABLE 3 — Driven CTQW-PRO (GPU)
# ═══════════════════════════════════════════════════════════════
print('\n[Table 3] Driven CTQW-PRO...')

all_t3 = {}
if gpu_ok:
    gpu_fns = build_gpu_methods(
        Apro_eigvals, Apro_eigvecs, _pro_src, _pro_dst, N, N_PRO,
        device=device, t=T_FIXED, n_steps=DRIVEN_N_STEPS, alpha=DRIVEN_ALPHA)

    def _build_psi(sidx_list):
        return build_psi_batch(sidx_list, N_PRO, device)

    # methods_list format: [(name, fn), ...] — exact từ notebook Cell 9
    methods_list = [
        ('t=0.1',          gpu_fns['ctqw_pro']),
        (f'driven_s{DRIVEN_N_STEPS}_a{DRIVEN_ALPHA}', gpu_fns['driven']),
    ]

    for label, dset in [('HMDB+CTD', eval_set1),
                         ('MarkerDB', eval_set2),
                         ('SMPDB',    eval_set3)]:
        t0  = time.time()
        res = run_driven_eval(
            dset, methods_list, node_idx, idx_pro, N, N_PRO,
            _build_psi, batch_size=32, label=label)
        all_t3[label] = res
        print(f'  {label}: {(time.time()-t0)/60:.1f} min')
else:
    # CPU fallback for driven (no GPU)
    for label, dset in [('HMDB+CTD', eval_set1),
                         ('MarkerDB', eval_set2),
                         ('SMPDB',    eval_set3)]:
        from methods import _ctqw_batch_raw
        def _driven_cpu(seeds, _n=N):
            from config import DRIVEN_N_STEPS as ns, DRIVEN_ALPHA as al
            valid_idx = [idx_pro[s] for s in seeds if s in idx_pro]
            if not valid_idx: return np.zeros(_n)
            psi_seed = np.zeros(N_PRO, dtype=complex)
            norm = 1.0 / np.sqrt(len(valid_idx))
            for idx in valid_idx: psi_seed[idx] = norm
            phases = np.exp(-1j * Apro_eigvals * T_FIXED)
            psi = psi_seed.copy()
            for _ in range(ns):
                coef   = Apro_eigvecs.conj().T @ psi
                walked = Apro_eigvecs @ (phases * coef)
                psi    = (1-al)*walked + al*psi_seed
                nrm = np.linalg.norm(psi)
                if nrm > 1e-9: psi = psi/nrm
            sc = np.zeros(_n)
            sc[_pro_dst] = (np.abs(psi)**2)[_pro_src]
            return sc
        df_dr = run_loo_eval(dset, _driven_cpu, node_idx, N, label=f'Driven/{label}')
        df_c  = all_t2[label]['t=0.1']
        all_t3[label] = {'t=0.1': df_c,
                          f'driven_s{DRIVEN_N_STEPS}_a{DRIVEN_ALPHA}': df_dr}


# ═══════════════════════════════════════════════════════════════
# TABLE 4 — NH γ=22 & RRF (G_pro)
# NH: Non-Hermitian walk — decay tại cofactor nodes
# RRF: Reciprocal Rank Fusion (NH + Driven)
# ═══════════════════════════════════════════════════════════════
print('\n[Table 4] NH γ=22 & RRF...')

# NH eigendecomp — chậm (~30s), cache lại
print(f'  Building NH eigendecomp (γ={NH_GAMMA})...', end=' ', flush=True)
t0_nh = time.time()
run_nh = make_nh_pro(
    A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
    RECON3D_COFACTORS, pro_nodes, NH_GAMMA, T_FIXED)
print(f'{time.time()-t0_nh:.1f}s')

# RRF = NH + Driven (CPU driven fn)
_driven_cpu_fn = lambda seeds: all_t3.get('SMPDB', {})   # placeholder
# Build CPU driven để dùng trong RRF (GPU driven không trả về numpy trực tiếp)
def _driven_cpu(seeds, _n=N):
    valid_idx = [idx_pro[s] for s in seeds if s in idx_pro]
    if not valid_idx: return np.zeros(_n)
    psi0 = np.zeros(N_PRO, dtype=complex)
    psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
    phases = np.exp(-1j * Apro_eigvals * T_FIXED)
    psi = psi0.copy()
    for _ in range(DRIVEN_N_STEPS):
        coef = Apro_eigvecs.conj().T @ psi
        psi  = (1 - DRIVEN_ALPHA) * (Apro_eigvecs @ (phases * coef)) + DRIVEN_ALPHA * psi0
        nrm  = np.linalg.norm(psi)
        if nrm > 1e-9: psi /= nrm
    sc = np.zeros(_n); sc[_pro_dst] = (np.abs(psi)**2)[_pro_src]
    return sc

run_rrf = make_rrf(run_nh, _driven_cpu, k=RRF_K)

all_t4 = {}
for label, dset in [('HMDB+CTD', eval_set1),
                     ('MarkerDB', eval_set2),
                     ('SMPDB',    eval_set3)]:
    t0 = time.time()
    df_nh  = run_loo_eval(dset, run_nh,  node_idx, N, label=f'NH/{label}')
    df_rrf = run_loo_eval(dset, run_rrf, node_idx, N, label=f'RRF/{label}')
    all_t4[label] = {'CTQW-PRO': all_t2[label]['t=0.1'],
                     f'NH γ={NH_GAMMA}': df_nh, 'RRF': df_rrf}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

# ═══════════════════════════════════════════════════════════════
# STEP 5 — Print results
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*72)
print('TABLE 1: RWR vs CTQW (G_cc)')
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    print_results_table(all_t1[label], label, method_order=['RWR','CTQW'])

print('\n' + '='*72)
print('TABLE 2: PROFANCY vs CTQW-PRO (G_pro)')
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    print_results_table(all_t2[label], label,
                        method_order=['PROFANCY','t=0.1'])

print('\n' + '='*72)
print('TABLE 3: CTQW-PRO vs Driven CTQW-PRO (G_pro)')
driven_name = f'driven_s{DRIVEN_N_STEPS}_a{DRIVEN_ALPHA}'
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    print_results_table(all_t3[label], label,
                        method_order=['t=0.1', driven_name])

# ── Statistical analysis — exact từ notebook Cell 8 ──────────
print('\n' + '='*72)
print('WILCOXON: CTQW-PRO vs PROFANCY')
wx_rows = []
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    df_p = all_t2[label].get('PROFANCY')
    df_c = all_t2[label].get('t=0.1')
    if df_p is not None and df_c is not None:
        df_wx = wilcoxon_table(df_c, df_p, label,
                               method_a='CTQW-PRO', method_b='PROFANCY')
        if df_wx is not None: wx_rows.append(df_wx)

print('\nWILCOXON: Driven vs CTQW-PRO (SMPDB)')
df_c = all_t3.get('SMPDB', {}).get('t=0.1')
df_d = all_t3.get('SMPDB', {}).get(driven_name)
if df_c is not None and df_d is not None:
    df_wx2 = wilcoxon_table(df_d, df_c, 'SMPDB',
                             method_a='Driven', method_b='CTQW-PRO')
    if df_wx2 is not None: wx_rows.append(df_wx2)

# Bootstrap CI
print('\n--- Bootstrap 95% CI ---')
from config import RANDOM_SEED
import numpy as np
for met in ['auc','mrr','r@20']:
    print(f'\n  {met.upper()}:')
    for label in ['HMDB+CTD','MarkerDB','SMPDB']:
        for mname, df in all_t2[label].items():
            if df is None or df.empty: continue
            mean, lo, hi = bootstrap_ci(df, met)
            print(f'    {label:<10} {mname:<20} '
                  f'{mean:.4f} [{lo:.4f},{hi:.4f}]  n={len(df)}')

# Win counts
print('\n--- Win counts (CTQW-PRO vs PROFANCY, AUC) ---')
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    df_p = all_t2[label].get('PROFANCY')
    df_c = all_t2[label].get('t=0.1')
    if df_p is not None and df_c is not None:
        w = win_counts(df_c, df_p, 'auc')
        print(f'  {label:<10}: CTQW-PRO={w["CTQW-PRO"]}, '
              f'PROFANCY={w["PROFANCY"]}, tie={w["tie"]} '
              f'(n={w["n_shared"]})')

# ═══════════════════════════════════════════════════════════════
# STEP 6 — Save
# ═══════════════════════════════════════════════════════════════
print('\nSaving...')
all_rows = []
for tname, results in [('table1', all_t1),
                        ('table2', all_t2),
                        ('table3', all_t3),
                        ('table4', all_t4)]:
    for label, res in results.items():
        for mname, df in res.items():
            if df is None or df.empty: continue
            d = df.copy()
            d['method'] = mname; d['source'] = label; d['table'] = tname
            all_rows.append(d)

if all_rows:
    out = RESULTS_DIR / 'main_results.csv'
    pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
    print(f'  Saved: {out}')

if wx_rows:
    out2 = RESULTS_DIR / 'wilcoxon_results.csv'
    pd.concat(wx_rows, ignore_index=True).to_csv(out2, index=False)
    print(f'  Saved: {out2}')


print('\n' + '='*72)
print('TABLE 4: CTQW-PRO vs NH vs RRF (G_pro)')
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    print_results_table(all_t4[label], label,
                        method_order=['CTQW-PRO', f'NH γ={NH_GAMMA}', 'RRF'])

print('\nWILCOXON: NH & RRF vs CTQW-PRO')
for label in ['HMDB+CTD','MarkerDB','SMPDB']:
    df_c   = all_t2[label].get('t=0.1')
    df_nh  = all_t4[label].get(f'NH γ={NH_GAMMA}')
    df_rrf = all_t4[label].get('RRF')
    if df_c is None: continue
    if df_nh  is not None:
        wilcoxon_table(df_nh,  df_c, label, method_a=f'NH γ={NH_GAMMA}', method_b='CTQW-PRO')
    if df_rrf is not None:
        wilcoxon_table(df_rrf, df_c, label, method_a='RRF', method_b='CTQW-PRO')

print('Done.')