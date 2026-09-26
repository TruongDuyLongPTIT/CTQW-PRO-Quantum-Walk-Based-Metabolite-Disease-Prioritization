import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import (CACHE_DIR, T_CTQW, NH_GAMMA, RECON3D_CURRENCY_METABOLITE,)
from graph import (parse_recon3d, build_gcc, build_gpro,build_hmdb_to_recon_initial, augment_hmdb_to_recon,compute_eigendecomp,)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set,build_eval_set1, build_eval_set2, build_eval_set3,)
from methods import (make_rwr, make_profancy, make_ctqw_gcc, make_ctqw_pro, make_nh_pro,)
from evaluation import (run_loo_eval,wilcoxon_table, bootstrap_ci, win_counts,print_results_table,)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Build graph
print('='*60); print('STEP 1 — Build graph')

recon_data   = parse_recon3d()
G_cc, _, N, node_idx, A_cc, _ = build_gcc(recon_data)
met_info     = recon_data['met_info']
pathway_mets = recon_data['pathway_mets']

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

# Transition matrix G_pro
deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro        = A_pro / deg_pro_safe[:, None]

# Transition matrix G_cc — for RWR baseline
deg_cc_safe  = np.where(A_cc.sum(1) > 0, A_cc.sum(1), 1.0)
P_cc         = A_cc / deg_cc_safe[:, None]

print(f'  Graph: {N} nodes, {G_cc.number_of_edges()} edges')
print(f'  G_pro: {N_PRO} nodes ({sum(1 for nd in pro_nodes if nd.startswith("__PATH__"))} pathway), '
      f'{G_pro.number_of_edges()} edges')

# Build eval sets
print('\nSTEP 2 — Build eval sets')

hmdb_data        = parse_hmdb()
hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)

hmdb_to_recon = build_hmdb_to_recon_initial(met_info, node_idx)
n_ik, n_nm = augment_hmdb_to_recon(
    hmdb_to_recon, met_info, node_idx,
    hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
    hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'],
)
print(f'  hmdb_to_recon: +{n_ik} IK, +{n_nm} name → {len(hmdb_to_recon)} total')

CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)

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
# một danh sách metabolite, đặc biệt ở SMPDB) — dùng cho bảng dedup
def _dedup_diseases(eval_set):
    rep = {}
    for d, mets in eval_set.items():
        rep.setdefault(tuple(sorted(mets)), d)
    return set(rep.values())

dedup_diseases = {label: _dedup_diseases(dset) for label, dset in EVAL_SETS.items()}
for label, dedup in dedup_diseases.items():
    print(f'  {label}: {len(EVAL_SETS[label])} tên bệnh → {len(dedup)} seed-set độc lập')

# Cal Eigendecomposition
print('\nSTEP 3 — Eigendecomposition')
Apro_eigvals, Apro_eigvecs = compute_eigendecomp(
    A_pro, CACHE_DIR / 'gpro_eigdecomp.npz')
Acc_eigvals, Acc_eigvecs   = compute_eigendecomp(
    A_cc, CACHE_DIR / 'gcc_eigdecomp.npz')
print('  Done.')

# ----------------TABLE 1 — RWR vs CTQW on G_cc ----------------
print('\n' + '='*60)
print('[Table 1] RWR vs CTQW on G_cc...')

run_rwr_gcc  = make_rwr(P_cc, node_idx, N)
run_ctqw_gcc = make_ctqw_gcc(Acc_eigvals, Acc_eigvecs, node_idx, N)

all_t1 = {}
for label, dset in EVAL_SETS.items():
    t0 = time.time()
    df_rwr  = run_loo_eval(dset, run_rwr_gcc, node_idx, N, label=f'RWR/{label}')
    df_ctqw = run_loo_eval(dset, run_ctqw_gcc, node_idx, N, label=f'CTQW/{label}')
    all_t1[label] = {'RWR': df_rwr, 'CTQW': df_ctqw}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

# -----------TABLE 2 — PROFANCY vs CTQW-PRO on G_pro--------------
print('\n[Table 2] PROFANCY vs CTQW-PRO on G_pro...')

run_profancy = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
run_ctqw_pro = make_ctqw_pro(
    Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)

all_t2 = {}
for label, dset in EVAL_SETS.items():
    t0 = time.time()
    df_prof = run_loo_eval(dset, run_profancy, node_idx, N,
                           label=f'PROFANCY/{label}')
    df_ctqw = run_loo_eval(dset, run_ctqw_pro, node_idx, N,
                           label=f'CTQW-PRO/{label}')
    all_t2[label] = {'PROFANCY': df_prof, 'CTQW-PRO': df_ctqw}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

# ----------TABLE 3 — NH-CTQW-PRO--------------------------------
print('\n[Table 3] NH-CTQW-PRO...')

print(f'  Building NH eigendecomp (γ={NH_GAMMA})...', end=' ', flush=True)
t0_nh = time.time()
run_nh = make_nh_pro(
    A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
    RECON3D_CURRENCY_METABOLITE, pro_nodes, NH_GAMMA, T_CTQW)
print(f'{time.time()-t0_nh:.1f}s')

all_t3 = {}
for label, dset in EVAL_SETS.items():
    t0 = time.time()
    df_nh  = run_loo_eval(dset, run_nh, node_idx, N, label=f'NH/{label}')
    all_t3[label] = {'CTQW-PRO': all_t2[label]['CTQW-PRO'],
                     f'NH γ={NH_GAMMA}': df_nh}
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')


def _dedup_view(results_dict, label):
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
_print_full_and_dedup(all_t2, ['PROFANCY', 'CTQW-PRO'])

print('\n' + '='*72)
print('TABLE 3: CTQW-PRO vs NH-CTQW-PRO (G_pro)')
_print_full_and_dedup(all_t3, ['CTQW-PRO', f'NH γ={NH_GAMMA}'])

# ---- Statistical analysis ----------------
# Mỗi phương pháp so với baseline tương ứng
print('\n' + '='*72)
print('\nWILCOXON: CTQW vs RWR (G_cc)')
for label in EVAL_SETS:
    wilcoxon_table(all_t1[label]['CTQW'], all_t1[label]['RWR'], label,
                   method_a='CTQW', method_b='RWR')

print('\nWILCOXON: CTQW-PRO vs PROFANCY (G_pro)')
for label in EVAL_SETS:
    wilcoxon_table(all_t2[label]['CTQW-PRO'], all_t2[label]['PROFANCY'], label,
                   method_a='CTQW-PRO', method_b='PROFANCY')

print('\nWILCOXON: NH vs CTQW-PRO')
for label in EVAL_SETS:
    df_c  = all_t2[label].get('CTQW-PRO')
    df_nh = all_t3[label].get(f'NH γ={NH_GAMMA}')
    if df_c is not None and df_nh is not None:
        wilcoxon_table(df_nh, df_c, label,
                       method_a=f'NH γ={NH_GAMMA}', method_b='CTQW-PRO')

# Wilcoxon chéo đồ thị: tác động của nút pathway (G_cc → G_pro) trong từng họ phương pháp
for m_pro, m_cc in [('PROFANCY', 'RWR'), ('CTQW-PRO', 'CTQW')]:
    print(f'\nWILCOXON CHÉO ĐỒ THỊ: {m_pro} (Gpro) vs {m_cc} (Gcc)')
    for label in EVAL_SETS:
        wilcoxon_table(all_t2[label][m_pro], all_t1[label][m_cc], label,
                       method_a=m_pro, method_b=m_cc)

print('\n--- Bootstrap 95% CI ---')
_ci_sources = [
    ('PROFANCY',          lambda l: all_t2[l].get('PROFANCY')),
    ('CTQW-PRO',          lambda l: all_t2[l].get('CTQW-PRO')),
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

# Win counts — metrics: auc + mrr
_win_pairs = [
    ('CTQW',     lambda l: all_t1[l].get('CTQW'),
     'RWR',      lambda l: all_t1[l].get('RWR')),
    ('CTQW-PRO', lambda l: all_t2[l].get('CTQW-PRO'),
     'PROFANCY', lambda l: all_t2[l].get('PROFANCY')),
    (f'NH γ={NH_GAMMA}', lambda l: all_t3[l].get(f'NH γ={NH_GAMMA}'),
     'CTQW-PRO',         lambda l: all_t2[l].get('CTQW-PRO')),
]
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
