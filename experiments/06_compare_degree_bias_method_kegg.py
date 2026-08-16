import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CACHE_DIR, RESULTS_DIR, T_FIXED, NH_GAMMA_KEGG, KEGG_CURRENCY_METABOLITE
from kegg_graph import build_kegg_metabolism_data, parse_hmdb_with_kegg, build_hmdb_to_kegg
from graph import build_gpro, compute_eigendecomp, compute_coreness
from eval_sets import build_hmdb_lookups, build_CURRENCY_METABOLITE_set, build_eval_set1, build_eval_set3
from methods import (make_profancy, make_metaborank_lite_pro, make_ctqw_pro,
                     make_nh_pro, make_netcore_pro, make_dada_ec_pro)
from evaluation import run_loo_eval, wilcoxon_table, print_results_table

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATASETS = ['HMDB+CTD', 'SMPDB']

print('=' * 78); print('SETUP (KEGG)'); print('=' * 78)
G_cc, node_idx, N, _, pathway_mets = build_kegg_metabolism_data()

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro = A_pro / deg_pro_safe[:, None]

Apro_eigvals, Apro_eigvecs = compute_eigendecomp(A_pro, CACHE_DIR / 'gpro_kegg_eigdecomp.npz')

hmdb_data = parse_hmdb_with_kegg(); hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups = build_hmdb_lookups(hmdb_metabolites)
hmdb_to_kegg = build_hmdb_to_kegg(hmdb_metabolites, node_idx)
CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)

eval_set1, _ = build_eval_set1(hmdb_metabolites, hmdb_lookups, hmdb_to_kegg,
                               node_idx, CURRENCY_METABOLITE,
                               currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)
eval_set3 = build_eval_set3(hmdb_metabolites, hmdb_to_kegg, node_idx, CURRENCY_METABOLITE,
                            currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)
dset_by_label = {'HMDB+CTD': eval_set1, 'SMPDB': eval_set3}
print(f'  N = {N}, N_PRO = {N_PRO}')
print(f'  HMDB+CTD: {len(eval_set1)} benh | SMPDB: {len(eval_set3)} benh')

deg_arr  = deg_pro.astype(float)
met_mask = np.array([(nd in node_idx) for nd in pro_nodes])

print('\n' + '=' * 78); print('PHAN A: chan doan coreness (KEGG)'); print('=' * 78)
t0 = time.time()
core_pro = compute_coreness(G_pro, pro_nodes)
print(f'  compute_coreness: {time.time()-t0:.2f}s')

gap_arr   = deg_arr - core_pro
ratio_arr = core_pro / deg_pro_safe

cm_in_graph = [nd for nd in KEGG_CURRENCY_METABOLITE
               if nd in idx_pro and nd in node_idx]
cm_mask = np.zeros(N_PRO, dtype=bool)
for nd in cm_in_graph:
    cm_mask[idx_pro[nd]] = True
other_mask = met_mask & (~cm_mask)

print(f'  {len(cm_in_graph)}/{len(KEGG_CURRENCY_METABOLITE)} currency metabolite trong G_pro')
print(f'  coreness toan mang: {core_pro.min():.0f}..{core_pro.max():.0f}, TB {core_pro.mean():.2f}')
print(f'\n  {"Nhom":<24}{"n":>6}{"degree TB":>12}{"coreness TB":>13}{"(d-k) TB":>11}{"k/d TB":>9}')
rows_A = []
for name, m in [('Currency metabolite', cm_mask),
                ('Metabolite thuong',   other_mask),
                ('Pathway node (ao)',   ~met_mask)]:
    print(f'  {name:<24}{m.sum():>6}{deg_arr[m].mean():>12.1f}{core_pro[m].mean():>13.1f}'
          f'{gap_arr[m].mean():>11.1f}{ratio_arr[m].mean():>9.3f}')
    rows_A.append({'nhom': name, 'n': int(m.sum()), 'degree_TB': deg_arr[m].mean(),
                   'coreness_TB': core_pro[m].mean(), 'gap_TB': gap_arr[m].mean(),
                   'k_tren_d_TB': ratio_arr[m].mean()})

rho_dk, p_dk = spearmanr(deg_arr[met_mask], core_pro[met_mask])
print(f'\n  Spearman(degree, coreness) tren node metabolite: rho = {rho_dk:.4f} (p={p_dk:.3g})')

print('\n  Trong so trung binh moi bien the gan cho tung nhom:')
print(f'  {"bien the":<14}{"currency":>11}{"thuong":>10}{"ti le":>10}')
for nm, w in [('core',  core_pro),
              ('diff',  1.0 / (gap_arr + 1.0)),
              ('ratio', core_pro / deg_pro_safe)]:
    wc, wn = w[cm_mask].mean(), w[other_mask].mean()
    print(f'  {nm:<14}{wc:>11.4f}{wn:>10.4f}{wc/wn:>9.2f}x'
          f'   {"UU AI currency" if wc > wn else "PHAT currency"}')

pd.DataFrame(rows_A).to_csv(RESULTS_DIR / 'kegg_A_coreness.csv', index=False)

print('\n' + '=' * 78); print('XAY DUNG PHUONG PHAP (KEGG)'); print('=' * 78)

METHODS = {}
METHODS['PROFANCY']            = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
METHODS['DADA-EC']             = make_dada_ec_pro(P_pro, idx_pro, N, N_PRO,
                                                  _pro_src, _pro_dst)
for v in ['core', 'diff', 'ratio']:
    METHODS[f'NetCore-{v}']    = make_netcore_pro(A_pro, deg_pro, idx_pro, N, N_PRO,
                                                  _pro_src, _pro_dst,
                                                  variant=v, core_pro=core_pro)
METHODS['MetaboRank-lite-PRO'] = make_metaborank_lite_pro(P_pro, idx_pro, node_idx,
                                                          N, N_PRO)
METHODS['CTQW-PRO']            = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro,
                                               N, N_PRO, _pro_src, _pro_dst)
print(f'  Dung NH-CTQW-PRO (gamma={NH_GAMMA_KEGG})...', end=' ', flush=True)
t0 = time.time()
METHODS['NH-CTQW-PRO']         = make_nh_pro(A_pro, idx_pro, N, N_PRO,
                                             _pro_src, _pro_dst,
                                             KEGG_CURRENCY_METABOLITE, pro_nodes,
                                             NH_GAMMA_KEGG, T_FIXED)
print(f'{time.time()-t0:.1f}s')

ORDER = list(METHODS.keys())
CLASSICAL = [m for m in ORDER if m not in ('CTQW-PRO', 'NH-CTQW-PRO')]
OURS      = ['CTQW-PRO', 'NH-CTQW-PRO']
print('  Thu tu bao cao:', ' | '.join(ORDER))

print('\n' + '=' * 78); print('PHAN B: danh gia LOO (KEGG)'); print('=' * 78)
all_res = {}
for label in RUN_DATASETS:
    dset = dset_by_label[label]
    all_res[label] = {}
    for mname in ORDER:
        t0 = time.time()
        all_res[label][mname] = run_loo_eval(dset, METHODS[mname], node_idx, N,
                                             label=f'{mname}/{label}')
        print(f'  {label} | {mname:<20}: {(time.time()-t0)/60:.1f} min')
    print_results_table(all_res[label], label, method_order=ORDER)


def safe_wilcoxon(df_a, df_b, lbl, method_a, method_b):
    if df_a is None or df_b is None:
        print(f'  {lbl}: thieu ket qua -- bo qua'); return
    shared = sorted(set(df_a['disease']) & set(df_b['disease']))
    cols = ['auc', 'mrr', 'r@5', 'r@10', 'r@20']
    if len(shared) >= 5:
        ka = df_a.set_index('disease').loc[shared, cols]
        kb = df_b.set_index('disease').loc[shared, cols]
        if np.allclose(ka.values, kb.values):
            print(f'\n--- {lbl} ---\n  Trung khop tuyet doi tren ca {len(shared)} benh '
                  f'-- bo qua kiem dinh.')
            return
    try:
        wilcoxon_table(df_a, df_b, lbl, method_a=method_a, method_b=method_b)
    except ValueError as e:
        print(f'\n--- {lbl} ---\n  Bo qua (scipy): {e}')


print('\n' + '=' * 78); print('PHAN D: kiem dinh thong ke (KEGG)'); print('=' * 78)
for label in RUN_DATASETS:
    print(f'\n{"#"*78}\n# {label}: tung phuong phap vs PROFANCY\n{"#"*78}')
    for m in ORDER:
        if m == 'PROFANCY':
            continue
        safe_wilcoxon(all_res[label][m], all_res[label]['PROFANCY'],
                      f'{label}: {m} vs PROFANCY', m, 'PROFANCY')

    print(f'\n{"#"*78}\n# {label}: CTQW-PRO / NH-CTQW-PRO vs TUNG baseline\n{"#"*78}')
    for ours in OURS:
        for m in CLASSICAL:
            safe_wilcoxon(all_res[label][ours], all_res[label][m],
                          f'{label}: {ours} vs {m}', ours, m)

print('\n' + '=' * 78); print('PHAN E: bang tong hop (KEGG)'); print('=' * 78)
for label in RUN_DATASETS:
    print(f'\n=== {label} ===')
    print(f'{"Phuong phap":<22}{"AUC":>8}{"MRR":>8}{"R@5":>8}{"R@10":>8}{"R@20":>8}')
    print('-' * 62)
    out = []
    for m in ORDER:
        df = all_res[label][m]
        if df is None or df.empty:
            continue
        print(f'{m:<22}{df["auc"].mean():>8.4f}{df["mrr"].mean():>8.4f}'
              f'{df["r@5"].mean():>8.4f}{df["r@10"].mean():>8.4f}{df["r@20"].mean():>8.4f}')
        out.append({'phuong_phap': m, 'AUC': df['auc'].mean(), 'MRR': df['mrr'].mean(),
                    'R@5': df['r@5'].mean(), 'R@10': df['r@10'].mean(),
                    'R@20': df['r@20'].mean(), 'n': len(df)})
    pd.DataFrame(out).to_csv(RESULTS_DIR / f'kegg_E_bang_{label.replace("+","")}.csv',
                             index=False)
    print(f'  -> da luu kegg_E_bang_{label.replace("+","")}.csv')

print('\n' + '=' * 78); print('XONG'); print('=' * 78)
