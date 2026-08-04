import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from scipy.stats import wilcoxon, ttest_rel, spearmanr, rankdata
from config import CACHE_DIR, T_FIXED, NH_GAMMA, RECON3D_CURRENCY_METABOLITE
from graph import (parse_recon3d, build_gcc, build_gpro, build_hmdb_to_recon_initial, augment_hmdb_to_recon, compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set, build_eval_set1, build_eval_set3)
from methods import make_ctqw_pro, make_profancy, make_nh_pro

# ---------SETUP--------------
print('='*60)
print('Setup...')
recon_data   = parse_recon3d()
G_cc, _, N, node_idx, _, _ = build_gcc(recon_data)
met_info     = recon_data['met_info']
pathway_mets = recon_data['pathway_mets']

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

Apro_eigvals, Apro_eigvecs = compute_eigendecomp(
    A_pro, CACHE_DIR / 'gpro_eigdecomp.npz')

hmdb_data        = parse_hmdb()
hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)
hmdb_to_recon    = build_hmdb_to_recon_initial(met_info, node_idx)
augment_hmdb_to_recon(hmdb_to_recon, met_info, node_idx,
    hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
    hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'])
CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)

eval_set1, _ = build_eval_set1(
    hmdb_metabolites, hmdb_lookups, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)
eval_set3 = build_eval_set3(
    hmdb_metabolites, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)

print(f'  HMDB+CTD: {len(eval_set1)} diseases')
print(f'  SMPDB:    {len(eval_set3)} diseases')

deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro        = A_pro / deg_pro_safe[:, None]
run_profancy = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
run_ctqw     = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO,
                              _pro_src, _pro_dst)
ctqw_fn      = lambda seeds: run_ctqw(seeds, [T_FIXED])[T_FIXED]
run_nh       = make_nh_pro(A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
                            RECON3D_CURRENCY_METABOLITE, pro_nodes, NH_GAMMA, T_FIXED)

dset_by_label = {'HMDB+CTD': eval_set1, 'SMPDB': eval_set3}

# ------------EXP 1 — Currency-metabolite rank bias -----------------
print('\n' + '='*60)
print('EXP 1: Currency-metabolite rank bias (PROFANCY vs CTQW-PRO vs NH-CTQW-PRO)')

cm_in_graph = [nd for nd in RECON3D_CURRENCY_METABOLITE if nd in idx_pro and nd in node_idx]
print(f'  {len(cm_in_graph)}/{len(RECON3D_CURRENCY_METABOLITE)} currency metabolite có trong G_pro')

def percentile_ranks(score_vec, cand, cm_set):
    s = np.array([score_vec[node_idx[nd]] for nd in cand])
    order = np.argsort(-s)
    rank_of = {cand[order[i]]: i for i in range(len(cand))}
    return {c: 100.0 * rank_of[c] / len(cand) for c in cm_set if c in rank_of}

for label, eval_set in dset_by_label.items():
    rows = []
    # tương quan Spearman: degree vs RANK (rank cục bộ trong candidate pool từng bệnh,
    # rank 1 = score cao nhất — ĐÚNG định nghĩa rank trong evaluation.py::compute_metrics)
    degree_rank_pairs_prof = []
    degree_rank_pairs_ctqw = []
    degree_rank_pairs_nh   = []
    for d, seed_mets in eval_set.items():
        seed_nodes = [m for m in seed_mets if m in idx_pro]
        if len(seed_nodes) < 2: continue
        sc_prof = run_profancy(seed_nodes)
        sc_ctqw = ctqw_fn(seed_nodes)
        sc_nh   = run_nh(seed_nodes)
        cand = [nd for nd in pro_nodes if nd in node_idx and nd not in seed_nodes]

        pctl_prof = percentile_ranks(sc_prof, cand, cm_in_graph)
        pctl_ctqw = percentile_ranks(sc_ctqw, cand, cm_in_graph)
        if not pctl_prof: continue

        rows.append({
            'disease': d, 'seed_key': tuple(sorted(seed_mets)), 'n_seed': len(seed_nodes),
            'avg_pctl_cm_PROFANCY': np.mean(list(pctl_prof.values())),
            'avg_pctl_cm_CTQWPRO':  np.mean(list(pctl_ctqw.values())),
        })

        cand_scores_prof = np.array([sc_prof[node_idx[nd]] for nd in cand])
        cand_scores_ctqw = np.array([sc_ctqw[node_idx[nd]] for nd in cand])
        cand_scores_nh   = np.array([sc_nh[node_idx[nd]]   for nd in cand])
        cand_ranks_prof = rankdata(-cand_scores_prof, method='average')
        cand_ranks_ctqw = rankdata(-cand_scores_ctqw, method='average')
        cand_ranks_nh   = rankdata(-cand_scores_nh,   method='average')
        for i, nd in enumerate(cand):
            deg_nd = float(deg_pro[idx_pro[nd]])
            degree_rank_pairs_prof.append((deg_nd, float(cand_ranks_prof[i])))
            degree_rank_pairs_ctqw.append((deg_nd, float(cand_ranks_ctqw[i])))
            degree_rank_pairs_nh.append((deg_nd, float(cand_ranks_nh[i])))

    df_pctl = pd.DataFrame(rows).sort_values('avg_pctl_cm_PROFANCY', ascending=False)
    df_pctl_dedup = df_pctl.drop_duplicates(subset='seed_key', keep='first')
    print(f'\n[{label}] {len(df_pctl)} tên bệnh → {df_pctl["seed_key"].nunique()} seed-set độc lập')

    for sub_df, tag in [(df_pctl, 'FULL'), (df_pctl_dedup, 'DEDUP')]:
        print(f'\n[{label} | {tag}] percentile currency-metabolite trung bình '
              f'({len(sub_df)} bệnh/seed-set):')
        print(f'  PROFANCY : {sub_df["avg_pctl_cm_PROFANCY"].mean():.2f}%')
        print(f'  CTQW-PRO : {sub_df["avg_pctl_cm_CTQWPRO"].mean():.2f}%')
        _, p_wx = wilcoxon(sub_df['avg_pctl_cm_CTQWPRO'], sub_df['avg_pctl_cm_PROFANCY'])
        _, p_t  = ttest_rel(sub_df['avg_pctl_cm_CTQWPRO'], sub_df['avg_pctl_cm_PROFANCY'])
        print(f'  Wilcoxon p={p_wx:.4g}  |  paired t-test p={p_t:.4g}')

    print(f'\n[{label}] Spearman(degree, RANK) — rank cục bộ trong candidate pool từng bệnh:')
    for name, pairs in [('PROFANCY', degree_rank_pairs_prof),
                         ('CTQW-PRO', degree_rank_pairs_ctqw),
                         ('NH-CTQW-PRO', degree_rank_pairs_nh)]:
        degs_r, ranks_r = zip(*pairs)
        rho_r, p_r = spearmanr(degs_r, ranks_r)
        print(f'  {name:<12}: rho={rho_r:.4f}, p={p_r:.4g} (n={len(degs_r)})')
