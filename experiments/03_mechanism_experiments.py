import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from scipy.stats import wilcoxon, ttest_rel, spearmanr, rankdata
from config import CACHE_DIR, T_FIXED, NH_GAMMA, RECON3D_CURRENCY_METABOLITE
from graph import (parse_recon3d, build_gcc, build_gpro, build_hmdb_to_recon_initial, augment_hmdb_to_recon, compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set, build_eval_set1, build_eval_set3)
from methods import make_ctqw_pro, make_profancy, make_metaborank_lite_pro, make_nh_pro

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
run_mb       = make_metaborank_lite_pro(P_pro, idx_pro, node_idx, N, N_PRO)
run_ctqw     = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO,
                              _pro_src, _pro_dst)
run_nh       = make_nh_pro(A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
                            RECON3D_CURRENCY_METABOLITE, pro_nodes, NH_GAMMA, T_FIXED)

dset_by_label = {'HMDB+CTD': eval_set1, 'SMPDB': eval_set3}

# ------------EXP 1 — Currency-metabolite rank bias -----------------
print('\n' + '='*60)
print('EXP 1: Currency-metabolite rank bias')
print('  (PROFANCY vs MetaboRank-lite-PRO vs CTQW-PRO vs NH-CTQW-PRO)')

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
    degree_rank_pairs_mb   = []
    degree_rank_pairs_ctqw = []
    degree_rank_pairs_nh   = []
    for d, seed_mets in eval_set.items():
        seed_nodes = [m for m in seed_mets if m in idx_pro]
        if len(seed_nodes) < 2: continue
        sc_prof = run_profancy(seed_nodes)
        sc_mb   = run_mb(seed_nodes)
        sc_ctqw = run_ctqw(seed_nodes)
        sc_nh   = run_nh(seed_nodes)
        cand = [nd for nd in pro_nodes if nd in node_idx and nd not in seed_nodes]

        pctl_prof = percentile_ranks(sc_prof, cand, cm_in_graph)
        pctl_mb   = percentile_ranks(sc_mb,   cand, cm_in_graph)
        pctl_ctqw = percentile_ranks(sc_ctqw, cand, cm_in_graph)
        if not pctl_prof: continue

        rows.append({
            'disease': d, 'seed_key': tuple(sorted(seed_mets)), 'n_seed': len(seed_nodes),
            'avg_pctl_cm_PROFANCY':  np.mean(list(pctl_prof.values())),
            'avg_pctl_cm_MBLitePRO': np.mean(list(pctl_mb.values())) if pctl_mb else np.nan,
            'avg_pctl_cm_CTQWPRO':   np.mean(list(pctl_ctqw.values())),
        })

        cand_scores_prof = np.array([sc_prof[node_idx[nd]] for nd in cand])
        cand_scores_mb   = np.array([sc_mb[node_idx[nd]]   for nd in cand])
        cand_scores_ctqw = np.array([sc_ctqw[node_idx[nd]] for nd in cand])
        cand_scores_nh   = np.array([sc_nh[node_idx[nd]]   for nd in cand])
        cand_ranks_prof = rankdata(-cand_scores_prof, method='average')
        cand_ranks_mb   = rankdata(-cand_scores_mb,   method='average')
        cand_ranks_ctqw = rankdata(-cand_scores_ctqw, method='average')
        cand_ranks_nh   = rankdata(-cand_scores_nh,   method='average')
        for i, nd in enumerate(cand):
            deg_nd = float(deg_pro[idx_pro[nd]])
            degree_rank_pairs_prof.append((deg_nd, float(cand_ranks_prof[i])))
            degree_rank_pairs_mb.append((deg_nd, float(cand_ranks_mb[i])))
            degree_rank_pairs_ctqw.append((deg_nd, float(cand_ranks_ctqw[i])))
            degree_rank_pairs_nh.append((deg_nd, float(cand_ranks_nh[i])))

    df_pctl = pd.DataFrame(rows).sort_values('avg_pctl_cm_PROFANCY', ascending=False)
    df_pctl_dedup = df_pctl.drop_duplicates(subset='seed_key', keep='first')
    print(f'\n[{label}] {len(df_pctl)} tên bệnh → {df_pctl["seed_key"].nunique()} seed-set độc lập')

    for sub_df, tag in [(df_pctl, 'FULL'), (df_pctl_dedup, 'DEDUP')]:
        print(f'\n[{label} | {tag}] percentile currency-metabolite trung bình '
              f'({len(sub_df)} bệnh/seed-set):')
        print(f'  PROFANCY           : {sub_df["avg_pctl_cm_PROFANCY"].mean():.2f}%')
        print(f'  MetaboRank-lite-PRO: {sub_df["avg_pctl_cm_MBLitePRO"].mean():.2f}%')
        print(f'  CTQW-PRO           : {sub_df["avg_pctl_cm_CTQWPRO"].mean():.2f}%')

        _, p_wx_mb = wilcoxon(sub_df['avg_pctl_cm_MBLitePRO'], sub_df['avg_pctl_cm_PROFANCY'])
        _, p_t_mb  = ttest_rel(sub_df['avg_pctl_cm_MBLitePRO'], sub_df['avg_pctl_cm_PROFANCY'])
        print(f'  MetaboRank-lite-PRO vs PROFANCY: Wilcoxon p={p_wx_mb:.4g}  |  paired t-test p={p_t_mb:.4g}')

        _, p_wx_ctqw = wilcoxon(sub_df['avg_pctl_cm_CTQWPRO'], sub_df['avg_pctl_cm_PROFANCY'])
        _, p_t_ctqw  = ttest_rel(sub_df['avg_pctl_cm_CTQWPRO'], sub_df['avg_pctl_cm_PROFANCY'])
        print(f'  CTQW-PRO vs PROFANCY:           Wilcoxon p={p_wx_ctqw:.4g}  |  paired t-test p={p_t_ctqw:.4g}')

        _, p_wx_cm = wilcoxon(sub_df['avg_pctl_cm_CTQWPRO'], sub_df['avg_pctl_cm_MBLitePRO'])
        _, p_t_cm  = ttest_rel(sub_df['avg_pctl_cm_CTQWPRO'], sub_df['avg_pctl_cm_MBLitePRO'])
        print(f'  CTQW-PRO vs MetaboRank-lite-PRO: Wilcoxon p={p_wx_cm:.4g}  |  paired t-test p={p_t_cm:.4g}')

    print(f'\n[{label}] Spearman(degree, RANK) — rank cục bộ trong candidate pool từng bệnh:')
    for name, pairs in [('PROFANCY', degree_rank_pairs_prof),
                         ('MetaboRank-lite-PRO', degree_rank_pairs_mb),
                         ('CTQW-PRO', degree_rank_pairs_ctqw),
                         ('NH-CTQW-PRO', degree_rank_pairs_nh)]:
        degs_r, ranks_r = zip(*pairs)
        rho_r, p_r = spearmanr(degs_r, ranks_r)
        print(f'  {name:<20}: rho={rho_r:.4f}, p={p_r:.4g} (n={len(degs_r)})')

# ------------EXP 2 — Local interference ratio (Bảng 10) -----------------
print('\n' + '='*60)
print('EXP 2: Local interference ratio r tại 3 currency metabolite bậc cao nhất: H+, H2O, ATP')
top3_cm = ['h', 'h2o', 'atp']  # H+, H2O, ATP
print('  Top-3 currency metabolite:')
for nd in top3_cm:
    print(f'    {nd:<8} ({met_info.get(nd, {}).get("name", nd)}) bậc={deg_pro[idx_pro[nd]]:.0f}')
 
def psi_full(seed_nodes, t=T_FIXED):
    valid = [idx_pro[s] for s in seed_nodes if s in idx_pro]
    psi0 = np.zeros(N_PRO, dtype=complex)
    psi0[valid] = 1.0 / np.sqrt(len(valid))
    c      = Apro_eigvecs.conj().T @ psi0
    phases = np.exp(-1j * Apro_eigvals * t)
    return Apro_eigvecs @ (phases * c)
 
def local_interference_ratio(psi_t, target_node):
    # r_j = |sum_{k in N(j)} psi_k(t)| / sum_{k in N(j)} |psi_k(t)|
    nb_idx = [idx_pro[nb] for nb in G_pro.neighbors(target_node) if nb in idx_pro]
    psi_nb = psi_t[nb_idx]
    denom  = np.sum(np.abs(psi_nb))
    if denom < 1e-15: return None
    return float(np.abs(np.sum(psi_nb)) / denom)
 
# 6 bệnh dùng trong Bảng 5 của paper
DISEASES_TABLE5 = ['Sepsis', 'Diabetes mellitus type 2', 'Phenylketonuria', 'Hypertension', "Crohn's disease", 'Colorectal cancer']
 
rows_r = []
for dname in DISEASES_TABLE5:
    match = next((d for d in eval_set1 if d.strip().lower() == dname.lower()), None)
    seed_mets = eval_set1[match]
    psi_t = psi_full(seed_mets)
    r_vals = [r for cm in top3_cm
              for r in [local_interference_ratio(psi_t, cm)] if r is not None]
    if r_vals:
        rows_r.append({'disease': match, 'n_seed': len(seed_mets),
                        'r_mean': float(np.mean(r_vals))})
        for cm, r in zip(top3_cm, r_vals):
            rows_r[-1][f'r_{cm}'] = r
 
df_r5 = pd.DataFrame(rows_r).sort_values('n_seed')
print(f'\n  Bảng 5 — {len(df_r5)} bệnh:')
print(df_r5.to_string(index=False))
all_r = df_r5[[f'r_{cm}' for cm in top3_cm]].values.flatten()
print(f'\n  Trung bình {len(all_r)} cặp (bệnh, currency metabolite): '
      f'{all_r.mean():.3f} ± {all_r.std():.3f} '
      f'(min {all_r.min():.3f}, max {all_r.max():.3f})')