"""
03_mechanism_experiments.py — Mechanism experiments cho paper CTQW-PRO.

2 experiments (bằng chứng cơ chế, Mục 4.3):
  EXP 1 — Currency-metabolite rank bias ← PROFANCY bị chi phối bởi degree (Bằng chứng thứ nhất, Mục 4.3.1)
  EXP 2 — Dephasing walk                ← coherence is essential (Bằng chứng thứ tư, Mục 4.3.2)
"""
import sys, time
from pathlib import Path
_src = Path(__file__).resolve().parent.parent / 'src'
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, ttest_rel

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED, RANDOM_SEED,
                    RECON3D_CURRENCY_METABOLITE)
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon,
                   compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set,
                       build_eval_set1, build_eval_set3)
from methods import make_ctqw_pro, make_profancy
from evaluation import run_loo_eval, wilcoxon_table, compute_metrics

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════
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

_ph0 = np.exp(-1j * Apro_eigvals * T_FIXED)   # reused across EXPs

dset_by_label = {'HMDB+CTD': eval_set1, 'SMPDB': eval_set3}


# ═══════════════════════════════════════════════════════════════
# EXP 1 — Currency-metabolite rank bias (Bằng chứng thứ nhất, Mục 4.3.1 paper)
# Percentile rank trung bình của currency metabolite trong candidate set
# (PROFANCY vs CTQW-PRO), và tương quan Spearman giữa node degree và score
# PROFANCY — xác nhận trực tiếp "PROFANCY bị chi phối bởi degree".
# percentile 0% = xếp hạng cao nhất; 100% = thấp nhất.
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 1: Currency-metabolite rank bias (PROFANCY vs CTQW-PRO)')

cm_in_graph = [nd for nd in RECON3D_CURRENCY_METABOLITE if nd in idx_pro and nd in node_idx]
print(f'  {len(cm_in_graph)}/{len(RECON3D_CURRENCY_METABOLITE)} currency metabolite có trong G_pro')

def percentile_ranks(score_vec, cand, cm_set):
    """percentile=0 là hạng cao nhất. score_vec: kích thước N (không gian G_cc)."""
    s = np.array([score_vec[node_idx[nd]] for nd in cand])
    order = np.argsort(-s)
    rank_of = {cand[order[i]]: i for i in range(len(cand))}
    return {c: 100.0 * rank_of[c] / len(cand) for c in cm_set if c in rank_of}

for label, eval_set in dset_by_label.items():
    rows = []
    degree_score_pairs = []
    for d, seed_mets in eval_set.items():
        seed_nodes = [m for m in seed_mets if m in idx_pro]
        if len(seed_nodes) < 2: continue
        sc_prof = run_profancy(seed_nodes)
        sc_ctqw = ctqw_fn(seed_nodes)
        cand = [nd for nd in pro_nodes if nd in node_idx and nd not in seed_nodes]

        pctl_prof = percentile_ranks(sc_prof, cand, cm_in_graph)
        pctl_ctqw = percentile_ranks(sc_ctqw, cand, cm_in_graph)
        if not pctl_prof: continue

        rows.append({
            'disease': d, 'seed_key': tuple(sorted(seed_mets)), 'n_seed': len(seed_nodes),
            'avg_pctl_cm_PROFANCY': np.mean(list(pctl_prof.values())),
            'avg_pctl_cm_CTQWPRO':  np.mean(list(pctl_ctqw.values())),
        })
        for nd in cand:
            degree_score_pairs.append((float(deg_pro[idx_pro[nd]]), float(sc_prof[node_idx[nd]])))

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

    degs, scores = zip(*degree_score_pairs)
    rho, p = spearmanr(degs, scores)
    print(f'\n[{label}] Spearman(degree, PROFANCY score): rho={rho:.4f}, p={p:.4g} '
          f'(n={len(degs)})')


# ═══════════════════════════════════════════════════════════════
# EXP 2 — Dephasing walk (Bằng chứng thứ tư, Mục 4.3.2 paper)
# So sánh CTQW-PRO gốc (sigma=0) với gần mất hết pha (sigma=5.0), trên
# top-15 bệnh có delta_mrr (CTQW-PRO - PROFANCY) cao nhất mỗi bộ dữ liệu,
# SAU KHI khử trùng seed-set (nhiều tên bệnh có thể trỏ cùng 1 seed-set,
# đặc biệt ở SMPDB — nếu không khử trùng, "15 bệnh" có thể chỉ là vài
# seed-set độc lập bị lặp tên).
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 2: Dephasing walk (decoherence experiment)')

SIGMA_GRID    = [0.0, 5.0]   # 0.0 = CTQW-PRO gốc (sanity check), 5.0 = gần mất hết pha
N_MC_PER_FOLD = 100
N_TOP         = 15           # số bệnh tốt nhất mỗi bộ dữ liệu, SAU khi khử trùng seed-set

_rng = np.random.default_rng(RANDOM_SEED)

def make_ctqw_pro_dephased(sigma, n_mc=N_MC_PER_FOLD):
    """CTQW-PRO với nhiễu pha Gauss trên từng eigenmode. sigma=0 → CTQW-PRO gốc."""
    def run_dephased(seed_nodes, _n=N):
        valid = [idx_pro[s] for s in seed_nodes if s in idx_pro]
        if not valid:
            return np.zeros(_n)
        psi0 = np.zeros(N_PRO, dtype=complex)
        psi0[valid] = 1.0 / np.sqrt(len(valid))
        c = Apro_eigvecs.conj().T @ psi0
        if sigma == 0.0:
            probs = np.abs(Apro_eigvecs @ (_ph0 * c))**2
        else:
            probs = np.zeros(N_PRO)
            for _ in range(n_mc):
                noise = _rng.normal(0.0, sigma, size=N_PRO)
                probs += np.abs(Apro_eigvecs @ ((_ph0 * np.exp(1j * noise)) * c))**2
            probs /= n_mc
        sc = np.zeros(_n)
        sc[_pro_dst] = probs[_pro_src]
        return sc
    return run_dephased

def _seed_key(mets):
    return tuple(sorted(mets))

print('Tính delta_mrr (CTQW-PRO - PROFANCY), khử trùng seed-set, chọn top-15 mỗi bộ')
top_diseases  = {}
for label, dset in dset_by_label.items():
    t0 = time.time()
    df_prof = run_loo_eval(dset, run_profancy, node_idx, N, label=f'PROFANCY/{label}')
    df_cpro = run_loo_eval(dset, ctqw_fn,      node_idx, N, label=f'CTQW-PRO/{label}')
    merged = df_cpro.merge(df_prof, on='disease', suffixes=('_ctqwpro', '_profancy'))
    merged['delta_mrr'] = merged['mrr_ctqwpro'] - merged['mrr_profancy']
    merged['seed_key']  = merged['disease'].map(lambda d: _seed_key(dset[d]))

    n_before = len(merged)
    merged = merged.sort_values('delta_mrr', ascending=False)
    merged = merged.drop_duplicates(subset='seed_key', keep='first')
    print(f'  {label}: khử trùng seed-set — {n_before} bệnh → {len(merged)} seed-set độc lập '
          f'({(time.time()-t0)/60:.1f} min)')

    top_diseases[label] = merged.head(N_TOP)['disease'].tolist()
    print(f'  {label} top-{N_TOP} (đã khử trùng):')
    print(merged.head(N_TOP)[['disease', 'delta_mrr', 'mrr_ctqwpro', 'mrr_profancy']]
          .to_string(index=False))

print('\nChạy dephasing (sigma=0.0 và 5.0) trên các bệnh top đã chọn')
results = {}
for label, dset in dset_by_label.items():
    subset = {d: dset[d] for d in top_diseases[label]}
    results[label] = {}
    for sigma in SIGMA_GRID:
        t0 = time.time()
        df = run_loo_eval(subset, make_ctqw_pro_dephased(sigma), node_idx, N,
                          label=f'{label} sigma={sigma}')
        results[label][sigma] = df
        if df is not None and not df.empty:
            print(f'  {label} sigma={sigma}: {(time.time()-t0)/60:.1f} min, '
                  f"MRR={df['mrr'].mean():.4f} AUC={df['auc'].mean():.4f} "
                  f"R@20={df['r@20'].mean():.4f}")

print(f'\n{"="*72}\nBẢNG TỔNG HỢP — top {N_TOP} bệnh (đã khử trùng), sigma=0 vs sigma=5.0')
print(f'{"="*72}')
wx_rows_dephase = []
for label in dset_by_label:
    df0, df5 = results[label].get(0.0), results[label].get(5.0)
    print(f'\n--- {label} (n={N_TOP}) ---')
    if df0 is not None and not df0.empty:
        print(f"  sigma=0.0: AUC={df0['auc'].mean():.4f} MRR={df0['mrr'].mean():.4f} "
              f"R@20={df0['r@20'].mean():.4f}")
    if df5 is not None and not df5.empty:
        print(f"  sigma=5.0: AUC={df5['auc'].mean():.4f} MRR={df5['mrr'].mean():.4f} "
              f"R@20={df5['r@20'].mean():.4f}")
    if df0 is not None and df5 is not None and not df0.empty and not df5.empty:
        df_wx = wilcoxon_table(df5, df0, f'{label} (sigma=5.0 vs 0.0)',
                               method_a='sigma=5.0', method_b='sigma=0.0')
        if df_wx is not None: wx_rows_dephase.append(df_wx)

        merged2 = df5.merge(df0, on='disease', suffixes=('_s5', '_s0'))
        merged2['delta_mrr_dephase'] = merged2['mrr_s5'] - merged2['mrr_s0']
        print(f'\n  Chi tiết từng bệnh ({label}):')
        print(merged2[['disease', 'mrr_s0', 'mrr_s5', 'delta_mrr_dephase']]
              .sort_values('delta_mrr_dephase').to_string(index=False))

if wx_rows_dephase:
    out = RESULTS_DIR / 'dephasing_wilcoxon.csv'
    pd.concat(wx_rows_dephase, ignore_index=True).to_csv(out, index=False)
    print(f'\nSaved: {out}')

print('\nDone.')
