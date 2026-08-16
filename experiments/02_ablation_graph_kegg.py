import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import numpy as np
from config import CACHE_DIR, T_FIXED, KEGG_CURRENCY_METABOLITE
from kegg_graph import build_kegg_metabolism_data, parse_hmdb_with_kegg, build_hmdb_to_kegg
from graph import build_gpro, build_clean_gpro, compute_eigendecomp
from eval_sets import build_hmdb_lookups, build_CURRENCY_METABOLITE_set, build_eval_set1, build_eval_set2, build_eval_set3
from methods import make_profancy, make_ctqw_pro
from evaluation import run_loo_eval, wilcoxon_table, win_counts

CACHE_DIR.mkdir(parents=True, exist_ok=True)

print('Building graphs...')
G_cc, node_idx, N, _, pathway_mets = build_kegg_metabolism_data()

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

(G_pro_cln, pro_nodes_cln, N_PRO_CLN, idx_pro_cln,
 A_pro_cln, deg_pro_cln, _pro_src_cln, _pro_dst_cln,
 node_idx_cln, N_cln) = build_clean_gpro(
    G_pro, node_idx, pathway_mets, KEGG_CURRENCY_METABOLITE)

print(f'  G_pro: {N_PRO} → clean: {N_PRO_CLN} nodes')

deg_safe = np.where(A_pro.sum(1)>0, A_pro.sum(1), 1.0); P_pro = A_pro/deg_safe[:,None]
deg_safe_cln = np.where(A_pro_cln.sum(1)>0, A_pro_cln.sum(1), 1.0)
P_pro_cln    = A_pro_cln/deg_safe_cln[:,None]

print('Building eval sets...')
hmdb_data        = parse_hmdb_with_kegg()
hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)
hmdb_to_kegg     = build_hmdb_to_kegg(hmdb_metabolites, node_idx)
CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)
eval_set1, disease_canonical = build_eval_set1(
    hmdb_metabolites, hmdb_lookups, hmdb_to_kegg, node_idx, CURRENCY_METABOLITE,
    currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)
eval_set2 = build_eval_set2(
    hmdb_metabolites, hmdb_lookups, hmdb_to_kegg,
    node_idx, CURRENCY_METABOLITE, disease_canonical,
    currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)
eval_set3 = build_eval_set3(
    hmdb_metabolites, hmdb_to_kegg, node_idx, CURRENCY_METABOLITE,
    currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)

EVAL_SETS = {'HMDB+CTD': eval_set1, 'MarkerDB': eval_set2, 'SMPDB': eval_set3}

print('Eigendecomposition...')
eigvals_o, eigvecs_o = compute_eigendecomp(A_pro, CACHE_DIR/'gpro_kegg_eigdecomp.npz')
eigvals_c, eigvecs_c = compute_eigendecomp(A_pro_cln, CACHE_DIR/'gpro_kegg_clean_eigdecomp.npz')

run_prof_o  = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
run_ctqw_o  = make_ctqw_pro(eigvals_o, eigvecs_o, idx_pro, N, N_PRO, _pro_src, _pro_dst)
run_prof_c  = make_profancy(P_pro_cln, idx_pro_cln, node_idx_cln, N_cln, N_PRO_CLN)
run_ctqw_c  = make_ctqw_pro(eigvals_c, eigvecs_c, idx_pro_cln, N_cln, N_PRO_CLN,
                             _pro_src_cln, _pro_dst_cln)

_ctqw_o_fn  = lambda seeds: run_ctqw_o(seeds, T_FIXED)
_ctqw_c_fn  = lambda seeds: run_ctqw_c(seeds, T_FIXED)

print('\nRunning ablation...')
all_results = {}
for label, dset in EVAL_SETS.items():
    t0 = time.time()
    all_results[label] = {
        'PROFANCY_orig': run_loo_eval(dset, run_prof_o, node_idx, N,         label=f'PROF_o/{label}'),
        'CTQW_orig':     run_loo_eval(dset, _ctqw_o_fn, node_idx, N,         label=f'CTQW_o/{label}'),
        'PROFANCY_cln':  run_loo_eval(dset, run_prof_c, node_idx_cln, N_cln, label=f'PROF_c/{label}'),
        'CTQW_cln':      run_loo_eval(dset, _ctqw_c_fn, node_idx_cln, N_cln, label=f'CTQW_c/{label}'),
    }
    print(f'  {label}: {(time.time()-t0)/60:.1f} min')

def _f(df, m):
    return f'{df[m].mean():.4f}' if df is not None and not df.empty else 'N/A'


PAIRS = [('PROFANCY', 'PROFANCY_orig', 'PROFANCY_cln'),
         ('CTQW-PRO', 'CTQW_orig',     'CTQW_cln')]

print('\n'+'='*72+'\nABLATION: Original vs Clean G_pro_kegg')
for label in EVAL_SETS:
    res = all_results[label]
    print(f'\n=== {label} ===')
    print(f"{'Method':<22} {'AUC_orig':>10} {'MRR_orig':>10} {'AUC_cln':>10} {'MRR_cln':>10} {'ΔMRR':>8}")
    print('-'*70)
    for nm, ko, kc in PAIRS:
        df_o = res.get(ko); df_c = res.get(kc)
        if df_o is None or df_c is None: continue
        shared = sorted(set(df_o['disease']) & set(df_c['disease']))
        if not shared: continue
        delta = (df_c.set_index('disease').loc[shared,'mrr'].mean() -
                 df_o.set_index('disease').loc[shared,'mrr'].mean())
        print(f'{nm:<22} {_f(df_o,"auc"):>10} {_f(df_o,"mrr"):>10} '
              f'{_f(df_c,"auc"):>10} {_f(df_c,"mrr"):>10} {delta:>+8.4f}')


print('\n'+'='*72+'\nWILCOXON: gốc vs đã loại currency metabolite (ghép cặp theo bệnh)')
for label in EVAL_SETS:
    res = all_results[label]
    for nm, ko, kc in PAIRS:
        df_o, df_c = res.get(ko), res.get(kc)
        if df_o is None or df_c is None: continue
        wilcoxon_table(df_c, df_o, label, method_a=f'{nm}_clean', method_b=f'{nm}_orig')

print('\n'+'='*72+'\nWIN-COUNT: gốc vs đã loại currency metabolite (ghép cặp theo bệnh, theo MRR)')
print('='*72)
for label in EVAL_SETS:
    res = all_results[label]
    for nm, ko, kc in PAIRS:
        df_o, df_c = res.get(ko), res.get(kc)
        if df_o is None or df_c is None: continue
        wc = win_counts(df_c, df_o, metric='mrr', name_a='loại_thắng', name_b='gốc_thắng')
        print(f"[{label}] {nm:<10} loại thắng {wc['loại_thắng']:>3} | gốc thắng {wc['gốc_thắng']:>3} | "
              f"hoà {wc['tie']:>3} | tổng {wc['n_shared']} bệnh "
              f"({100*wc['loại_thắng']/wc['n_shared']:.1f}% loại thắng)")
