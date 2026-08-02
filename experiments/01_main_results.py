"""
01_main_results.py — Main experiment.
Table 1: CTQW vs RWR (G_cc)    [chạy trước]
Table 2: CTQW-PRO vs PROFANCY (G_pro)
Table 3: NH-CTQW-PRO vs CTQW-PRO

+ Wilcoxon chéo đồ thị: PROFANCY(Gpro) vs RWR(Gcc), CTQW-PRO(Gpro) vs CTQW(Gcc)
  — bằng chứng thống kê cho "augment pathway node giúp CTQW nhiều hơn RWR".
+ Bảng dedup (khử seed-set trùng lặp, nhiều tên bệnh cùng trỏ 1 seed-set) cho
  Table 1/2/3, song song bảng full.

Closely follows notebook Cell 7 (LOO) and Cell 8 (stats).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, CACHE_DIR,
    T_FIXED, NH_GAMMA,
    RECON3D_CURRENCY_METABOLITE,
)
from graph import (
    parse_recon3d, build_gcc, build_gpro,
    build_hmdb_to_recon_initial, augment_hmdb_to_recon,
    compute_eigendecomp,
)
from eval_sets import (
    parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set,
    build_eval_set1, build_eval_set2, build_eval_set3,
)
from methods import (
    run_rwr, make_profancy, make_ctqw_pro, make_ctqw_gcc,
    make_nh_pro,
)
from evaluation import (
    run_loo_eval,
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
G_cc, _, N, node_idx, A_cc, _ = build_gcc(recon_data)
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

# CURRENCY_METABOLITE name-based set (exact từ notebook Cell 3)
CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)

# Build eval sets
eval_set1, disease_canonical = build_eval_set1(
    hmdb_metabolites, hmdb_lookups, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)
eval_set2 = build_eval_set2(
    hmdb_metabolites, hmdb_lookups, hmdb_to_recon,
    node_idx, CURRENCY_METABOLITE, disease_canonical)
eval_set3 = build_eval_set3(
    hmdb_metabolites, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)

print(f'  eval_set1 (HMDB+CTD): {len(eval_set1)} diseases')
print(f'  eval_set2 (MarkerDB): {len(eval_set2)} diseases')
print(f'  eval_set3 (SMPDB):    {len(eval_set3)} diseases')

EVAL_SETS = {'HMDB+CTD': eval_set1, 'MarkerDB': eval_set2, 'SMPDB': eval_set3}

# Bệnh đại diện sau khi khử trùng seed-set (nhiều tên bệnh có thể trỏ tới cùng
# một danh sách metabolite, đặc biệt ở SMPDB) — dùng cho bảng dedup ở STEP 5.
def _dedup_diseases(eval_set):
    rep = {}
    for d, mets in eval_set.items():
        rep.setdefault(tuple(sorted(mets)), d)
    return set(rep.values())

dedup_diseases = {label: _dedup_diseases(dset) for label, dset in EVAL_SETS.items()}
for label, dedup in dedup_diseases.items():
    print(f'  {label}: {len(EVAL_SETS[label])} tên bệnh → {len(dedup)} seed-set độc lập')

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
# TABLE 1 — RWR vs CTQW on G_cc  [FIRST: simpler, establishes baseline]
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('[Table 1] RWR vs CTQW on G_cc...')

# G_cc methods — closures capturing N at definition time
_rwr_fn   = lambda seeds: run_rwr(seeds, P_cc, node_idx, N)
_ctqw_gcc = make_ctqw_gcc(Acc_eigvals, Acc_eigvecs, N)
_ctqw_gcc_fn = lambda seeds: _ctqw_gcc(seeds, node_idx)

all_t1 = {}
for label, dset in EVAL_SETS.items():
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
for label, dset in EVAL_SETS.items():
    t0 = time.time()
    df_prof = run_loo_eval(dset, run_profancy, node_idx, N,
                           label=f'PROFANCY/{label}')
    df_ctqw = run_loo_eval(dset, _ctqw_pro_fn, node_idx, N,
                           label=f'CTQW-PRO/{label}')
    all_t2[label] = {'PROFANCY': df_prof, 't=0.1': df_ctqw}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

# ═══════════════════════════════════════════════════════════════
# TABLE 3 — NH-CTQW-PRO
# ═══════════════════════════════════════════════════════════════
print('\n[Table 3] NH-CTQW-PRO...')

# NH eigendecomp — chậm (~30s), cache lại
print(f'  Building NH eigendecomp (γ={NH_GAMMA})...', end=' ', flush=True)
t0_nh = time.time()
run_nh = make_nh_pro(
    A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
    RECON3D_CURRENCY_METABOLITE, pro_nodes, NH_GAMMA, T_FIXED)
print(f'{time.time()-t0_nh:.1f}s')

all_t3 = {}
for label, dset in EVAL_SETS.items():
    t0 = time.time()
    df_nh  = run_loo_eval(dset, run_nh, node_idx, N, label=f'NH/{label}')
    all_t3[label] = {'CTQW-PRO': all_t2[label]['t=0.1'],
                     f'NH γ={NH_GAMMA}': df_nh}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Print results (full + dedup cho Table 1–3)
# ═══════════════════════════════════════════════════════════════
def _dedup_view(results_dict, label):
    """Lọc mỗi df trong results_dict về đúng các bệnh đại diện (đã khử trùng seed-set)."""
    keep = dedup_diseases[label]
    return {m: (df[df['disease'].isin(keep)] if df is not None else df)
            for m, df in results_dict.items()}

def _print_full_and_dedup(all_t, method_order):
    for label in EVAL_SETS:
        print_results_table(all_t[label], label, method_order=method_order)
        print_results_table(_dedup_view(all_t[label], label),
                            f'{label} (dedup)', method_order=method_order)

print('\n' + '='*72)
print('TABLE 1: RWR vs CTQW (G_cc)')
_print_full_and_dedup(all_t1, ['RWR', 'CTQW'])

print('\n' + '='*72)
print('TABLE 2: PROFANCY vs CTQW-PRO (G_pro)')
_print_full_and_dedup(all_t2, ['PROFANCY', 't=0.1'])

print('\n' + '='*72)
print('TABLE 3: CTQW-PRO vs NH-CTQW-PRO (G_pro)')
_print_full_and_dedup(all_t3, ['CTQW-PRO', f'NH γ={NH_GAMMA}'])

# ── Statistical analysis ──────────────────────────────────────
print('\n' + '='*72)
print('WILCOXON: CTQW vs RWR')
wx_rows = []
for label in EVAL_SETS:
    df_rwr  = all_t1[label].get('RWR')
    df_ctqw = all_t1[label].get('CTQW')
    if df_rwr is not None and df_ctqw is not None:
        df_wx0 = wilcoxon_table(df_ctqw, df_rwr, label,
                                method_a='CTQW', method_b='RWR')
        if df_wx0 is not None: wx_rows.append(df_wx0)

print('\nWILCOXON: CTQW-PRO vs PROFANCY')
for label in EVAL_SETS:
    df_p = all_t2[label].get('PROFANCY')
    df_c = all_t2[label].get('t=0.1')
    if df_p is not None and df_c is not None:
        df_wx = wilcoxon_table(df_c, df_p, label,
                               method_a='CTQW-PRO', method_b='PROFANCY')
        if df_wx is not None: wx_rows.append(df_wx)

print('\nWILCOXON: NH vs CTQW-PRO')
for label in EVAL_SETS:
    df_c  = all_t2[label].get('t=0.1')
    df_nh = all_t3[label].get(f'NH γ={NH_GAMMA}')
    if df_c is not None and df_nh is not None:
        df_wx2 = wilcoxon_table(df_nh, df_c, label,
                                method_a=f'NH γ={NH_GAMMA}', method_b='CTQW-PRO')
        if df_wx2 is not None: wx_rows.append(df_wx2)

# Chéo đồ thị: augment pathway node giúp CTQW nhiều hơn RWR? (cùng họ phương
# pháp, Gpro vs Gcc — Bảng 1/2 chỉ kiểm định trong-cùng-đồ-thị, đây là phần
# còn thiếu để so PROFANCY[=RWR/Gpro] với RWR[Gcc], và CTQW-PRO[Gpro] với CTQW[Gcc])
print('\nWILCOXON CHÉO ĐỒ THỊ: PROFANCY (Gpro) vs RWR (Gcc)')
for label in EVAL_SETS:
    df_wx5 = wilcoxon_table(all_t2[label]['PROFANCY'], all_t1[label]['RWR'], label,
                            method_a='PROFANCY', method_b='RWR')
    if df_wx5 is not None: wx_rows.append(df_wx5)

print('\nWILCOXON CHÉO ĐỒ THỊ: CTQW-PRO (Gpro) vs CTQW (Gcc)')
for label in EVAL_SETS:
    df_wx6 = wilcoxon_table(all_t2[label]['t=0.1'], all_t1[label]['CTQW'], label,
                            method_a='CTQW-PRO', method_b='CTQW')
    if df_wx6 is not None: wx_rows.append(df_wx6)

# Bootstrap CI — tất cả methods
print('\n--- Bootstrap 95% CI ---')
_ci_sources = [
    ('PROFANCY',          lambda l: all_t2[l].get('PROFANCY')),
    ('CTQW-PRO',          lambda l: all_t2[l].get('t=0.1')),
    (f'NH γ={NH_GAMMA}', lambda l: all_t3[l].get(f'NH γ={NH_GAMMA}')),
]
for met in ['auc','mrr','r@20']:
    print(f'\n  {met.upper()}:')
    for mname, get_df in _ci_sources:
        for label in EVAL_SETS:
            df = get_df(label)
            if df is None or df.empty: continue
            mean, lo, hi = bootstrap_ci(df, met)
            print(f'    {label:<10} {mname:<20} '
                  f'{mean:.4f} [{lo:.4f},{hi:.4f}]  n={len(df)}')

# Win counts — all main method pairs, metrics: auc + mrr
_win_pairs = [
    ('CTQW',    lambda l: all_t1[l].get('CTQW'),
     'RWR',     lambda l: all_t1[l].get('RWR')),
    ('CTQW-PRO',lambda l: all_t2[l].get('t=0.1'),
     'PROFANCY', lambda l: all_t2[l].get('PROFANCY')),
    (f'NH γ={NH_GAMMA}', lambda l: all_t3[l].get(f'NH γ={NH_GAMMA}'),
     'CTQW-PRO',         lambda l: all_t2[l].get('t=0.1')),
]
wc_rows = []
print('\n--- Win counts ---')
print(f"  {'Pair':<28} {'Dataset':<10} {'Metric':<6} "
      f"{'A wins':>7} {'B wins':>7} {'tie':>5} {'n':>5}")
print('  ' + '-'*68)
for name_a, get_a, name_b, get_b in _win_pairs:
    for metric in ['auc', 'mrr']:
        for label in EVAL_SETS:
            df_a = get_a(label); df_b = get_b(label)
            if df_a is None or df_b is None: continue
            w = win_counts(df_a, df_b, metric, name_a=name_a, name_b=name_b)
            pair_str = f'{name_a} vs {name_b}'
            print(f'  {pair_str:<28} {label:<10} {metric:<6} '
                  f'{w[name_a]:>7} {w[name_b]:>7} {w["tie"]:>5} {w["n_shared"]:>5}')
            wc_rows.append({
                'name_a': name_a, 'name_b': name_b,
                'source': label,  'metric': metric,
                'wins_a': w[name_a], 'wins_b': w[name_b],
                'tie':    w['tie'],  'n':      w['n_shared'],
            })

# ═══════════════════════════════════════════════════════════════
# STEP 6 — Save
# ═══════════════════════════════════════════════════════════════
print('\nSaving...')
all_rows = []
# CTQW-PRO trong all_t3 là alias của all_t2 → skip để tránh duplicate
_skip = {'table3': {'CTQW-PRO'}}
for tname, results in [('table1', all_t1),
                        ('table2', all_t2),
                        ('table3', all_t3)]:
    for label, res in results.items():
        for mname, df in res.items():
            if df is None or df.empty: continue
            if mname in _skip.get(tname, set()): continue
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

if wc_rows:
    out3 = RESULTS_DIR / 'win_counts.csv'
    pd.DataFrame(wc_rows).to_csv(out3, index=False)
    print(f'  Saved: {out3}')


print('Done.')
