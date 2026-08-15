import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, spearmanr, rankdata

from config import (CACHE_DIR, RESULTS_DIR, T_FIXED, NH_GAMMA, RWR_R,
                    RECON3D_CURRENCY_METABOLITE)
from graph import (parse_recon3d, build_gcc, build_gpro, compute_eigendecomp,
                   compute_coreness,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set,
                       build_eval_set1, build_eval_set3)
from methods import (make_profancy, make_metaborank_lite_pro, make_ctqw_pro,
                     make_nh_pro, make_netcore_pro, make_dada_ec_pro)
from evaluation import run_loo_eval, wilcoxon_table, print_results_table

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- cau hinh ----
RUN_DATASETS    = ['HMDB+CTD', 'SMPDB']    # bo MarkerDB (n=23, kiem dinh yeu)
RUN_SENSITIVITY = True                     # Phan F: tham so goc cua tac gia
NETCORE_ALPHA   = 0.8                      # restart mac dinh trong bai NetCore
DADA_R          = 0.3                      # restart dung trong bai DADA

# ---------------SETUP--------------------
print('=' * 78); print('SETUP'); print('=' * 78)
recon_data = parse_recon3d()
G_cc, _, N, node_idx, _, _ = build_gcc(recon_data)
met_info, pathway_mets = recon_data['met_info'], recon_data['pathway_mets']

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro = A_pro / deg_pro_safe[:, None]

Apro_eigvals, Apro_eigvecs = compute_eigendecomp(A_pro, CACHE_DIR / 'gpro_eigdecomp.npz')

hmdb_data = parse_hmdb(); hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups = build_hmdb_lookups(hmdb_metabolites)
hmdb_to_recon = build_hmdb_to_recon_initial(met_info, node_idx)
augment_hmdb_to_recon(hmdb_to_recon, met_info, node_idx,
    hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
    hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'])
CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)

eval_set1, _ = build_eval_set1(hmdb_metabolites, hmdb_lookups, hmdb_to_recon,
                               node_idx, CURRENCY_METABOLITE)
eval_set3 = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)
dset_by_label = {'HMDB+CTD': eval_set1, 'SMPDB': eval_set3}
print(f'  N = {N}, N_PRO = {N_PRO}')
print(f'  HMDB+CTD: {len(eval_set1)} benh | SMPDB: {len(eval_set3)} benh')

deg_arr  = deg_pro.astype(float)
met_mask = np.array([(nd in node_idx) for nd in pro_nodes])

# --- CHAN DOAN CORENESS--------------

print('\n' + '=' * 78); print('PHAN A: chan doan coreness'); print('=' * 78)
t0 = time.time()
core_pro = compute_coreness(G_pro, pro_nodes)
print(f'  compute_coreness: {time.time()-t0:.2f}s')

gap_arr   = deg_arr - core_pro
ratio_arr = core_pro / deg_pro_safe

cm_in_graph = [nd for nd in RECON3D_CURRENCY_METABOLITE
               if nd in idx_pro and nd in node_idx]
cm_mask = np.zeros(N_PRO, dtype=bool)
for nd in cm_in_graph:
    cm_mask[idx_pro[nd]] = True
other_mask = met_mask & (~cm_mask)

print(f'  {len(cm_in_graph)}/{len(RECON3D_CURRENCY_METABOLITE)} currency metabolite trong G_pro')
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
print('  -> rho cang gan 1 thi coreness cang chi la ban sao cua degree, va chuan hoa')
print('     theo coreness (NetCore-core) cang KHUECH DAI thien lech thay vi giam.')

# trong so ma tung bien the gan cho currency metabolite vs metabolite thuong
print(f'\n  Trong so trung binh moi bien the gan cho tung nhom:')
print(f'  {"bien the":<14}{"currency":>11}{"thuong":>10}{"ti le":>10}')
for nm, w in [('core',  core_pro),
              ('diff',  1.0 / (gap_arr + 1.0)),
              ('ratio', core_pro / deg_pro_safe)]:
    wc, wn = w[cm_mask].mean(), w[other_mask].mean()
    print(f'  {nm:<14}{wc:>11.4f}{wn:>10.4f}{wc/wn:>9.2f}x'
          f'   {"UU AI currency" if wc > wn else "PHAT currency"}')

pd.DataFrame(rows_A).to_csv(RESULTS_DIR / 'sec46_A_coreness.csv', index=False)

# -------- cac method can so sánh--------------
print('\n' + '=' * 78); print('XAY DUNG PHUONG PHAP'); print('=' * 78)

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
print(f'  Dung NH-CTQW-PRO (gamma={NH_GAMMA})...', end=' ', flush=True)
t0 = time.time()
METHODS['NH-CTQW-PRO']         = make_nh_pro(A_pro, idx_pro, N, N_PRO,
                                             _pro_src, _pro_dst,
                                             RECON3D_CURRENCY_METABOLITE, pro_nodes,
                                             NH_GAMMA, T_FIXED)
print(f'{time.time()-t0:.1f}s')

ORDER    = list(METHODS.keys())              # thu tu hien thi da chot
CLASSICAL = [m for m in ORDER if m not in ('CTQW-PRO', 'NH-CTQW-PRO')]
OURS      = ['CTQW-PRO', 'NH-CTQW-PRO']
print('  Thu tu bao cao:', ' | '.join(ORDER))

# ------------ LOO-------------
print('\n' + '=' * 78); print('PHAN B: danh gia LOO'); print('=' * 78)
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

# -----------DO DEGREE BIAS------------
print('\n' + '=' * 78); print('PHAN C: do degree bias'); print('=' * 78)


def percentile_ranks(score_vec, cand, cm_set):
    s = np.array([score_vec[node_idx[nd]] for nd in cand])
    order = np.argsort(-s)
    rank_of = {cand[order[i]]: i for i in range(len(cand))}
    return {c: 100.0 * rank_of[c] / len(cand) for c in cm_set if c in rank_of}


bias = {}          # bias[label][method] = (pctl_full, pctl_dedup, rho)
pctl_df = {}
for label in RUN_DATASETS:
    eval_set = dset_by_label[label]
    rows = []
    deg_chunks  = []                       # bac cua ung vien: GIONG NHAU cho moi
                                           # phuong phap -> chi luu MOT ban
    rank_chunks = {m: [] for m in ORDER}   # thu hang: khac nhau tung phuong phap

    t0 = time.time()
    for d_name, seed_mets in eval_set.items():
        seed_nodes = [m for m in seed_mets if m in idx_pro]
        if len(seed_nodes) < 2:
            continue
        sc = {m: METHODS[m](seed_nodes) for m in ORDER}
        seed_set = set(seed_nodes)
        cand = [nd for nd in pro_nodes if nd in node_idx and nd not in seed_set]

        pctl = {m: percentile_ranks(sc[m], cand, cm_in_graph) for m in ORDER}
        if not pctl['PROFANCY']:
            continue
        row = {'disease': d_name, 'seed_key': tuple(sorted(seed_mets))}
        for m in ORDER:
            row[m] = np.mean(list(pctl[m].values())) if pctl[m] else np.nan
        rows.append(row)

        cand_ni  = np.fromiter((node_idx[nd] for nd in cand), dtype=np.intp, count=len(cand))
        cand_deg = np.fromiter((deg_pro[idx_pro[nd]] for nd in cand),
                               dtype=np.float64, count=len(cand))
        deg_chunks.append(cand_deg)
        for m in ORDER:
            rank_chunks[m].append(rankdata(-sc[m][cand_ni], method='average'))

    if not rows:
        print(f'\n[{label}] KHONG co benh nao hop le -- bo qua bo du lieu nay.')
        bias[label] = {m: (np.nan, np.nan, np.nan) for m in ORDER}
        continue

    df = pd.DataFrame(rows)
    df_dedup = df.drop_duplicates(subset='seed_key', keep='first')
    pctl_df[label] = df
    deg_all = np.concatenate(deg_chunks)          # dung chung cho moi phuong phap
    print(f'\n[{label}] {len(df)} ten benh -> {df["seed_key"].nunique()} seed-set doc lap'
          f'  ({(time.time()-t0)/60:.1f} min)')

    print(f'\n  {"phuong phap":<22}{"pctl CM (full)":>16}{"pctl CM (dedup)":>17}'
          f'{"rho(deg,rank)":>15}')
    bias[label] = {}
    for m in ORDER:
        rr = np.concatenate(rank_chunks[m])
        rho, _ = spearmanr(deg_all, rr)
        bias[label][m] = (df[m].mean(), df_dedup[m].mean(), rho)
        print(f'  {m:<22}{df[m].mean():>15.2f}%{df_dedup[m].mean():>16.2f}%{rho:>15.4f}')
    print('  (pctl CANG CAO cang it thien lech; rho cang GAN 0 cang it thien lech,')
    print('   rho am nghia la node bac cao duoc xep hang cao)')

    print(f'\n  Kiem dinh percentile CM so voi PROFANCY (paired theo benh):')
    for m in ORDER:
        if m == 'PROFANCY':
            continue
        a, b = df[m].values, df['PROFANCY'].values
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
            print(f'    {m:<22} khong du du lieu / khong khac biet')
            continue
        _, p_wx = wilcoxon(a[ok], b[ok]); _, p_tt = ttest_rel(a[ok], b[ok])
        print(f'    {m:<22} Wilcoxon p={p_wx:.4g} | paired t-test p={p_tt:.4g}')

    df.to_csv(RESULTS_DIR / f'sec46_C_bias_{label.replace("+","")}.csv', index=False)

# -------------KIEM DINH THONG KE------------
print('\n' + '=' * 78); print('PHAN D: kiem dinh thong ke'); print('=' * 78)


def safe_wilcoxon(df_a, df_b, lbl, method_a, method_b):
    """scipy.wilcoxon tra ve nan neu x-y = 0 o moi phan tu (2 phuong phap
    trung khop tuyet doi). Chan truoc de in thong bao ro rang."""
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

# ------------ BANG TONG HOP CHO MUC 4.6----------
print('\n' + '=' * 78); print('PHAN E: bang tong hop (dang dua vao Muc 4.6)'); print('=' * 78)
for label in RUN_DATASETS:
    print(f'\n=== {label} ===')
    print(f'{"Phuong phap":<22}{"AUC":>8}{"MRR":>8}{"R@5":>8}{"R@10":>8}{"R@20":>8}'
          f'{"pctl CM":>10}{"rho":>9}')
    print('-' * 81)
    out = []
    for m in ORDER:
        df = all_res[label][m]
        if df is None or df.empty:
            continue
        p_full, p_ded, rho = bias[label][m]
        print(f'{m:<22}{df["auc"].mean():>8.4f}{df["mrr"].mean():>8.4f}'
              f'{df["r@5"].mean():>8.4f}{df["r@10"].mean():>8.4f}{df["r@20"].mean():>8.4f}'
              f'{p_full:>9.2f}%{rho:>9.4f}')
        out.append({'phuong_phap': m, 'AUC': df['auc'].mean(), 'MRR': df['mrr'].mean(),
                    'R@5': df['r@5'].mean(), 'R@10': df['r@10'].mean(),
                    'R@20': df['r@20'].mean(), 'pctl_CM_full': p_full,
                    'pctl_CM_dedup': p_ded, 'rho_deg_rank': rho, 'n': len(df)})
    pd.DataFrame(out).to_csv(RESULTS_DIR / f'sec46_E_bang_{label.replace("+","")}.csv',
                             index=False)
    print(f'  -> da luu sec46_E_bang_{label.replace("+","")}.csv')

# ------------ DO NHAY THAM SO (tham so goc cua tac gia)---------
if RUN_SENSITIVITY:
    print('\n' + '=' * 78)
    print('PHAN F: do nhay tham so -- chay lai voi THAM SO GOC cua tac gia')
    print(f'  NetCore: alpha={NETCORE_ALPHA} (bai goc)  vs  {RWR_R} (dung o tren)')
    print(f'  DADA   : r={DADA_R} (bai goc)           vs  {RWR_R} (dung o tren)')
    print('=' * 78)

    SENS = {}
    SENS[f'PROFANCY a={NETCORE_ALPHA}'] = make_profancy(P_pro, idx_pro, node_idx,
                                                        N, N_PRO, r=NETCORE_ALPHA)
    for v in ['core', 'diff', 'ratio']:
        SENS[f'NetCore-{v} a={NETCORE_ALPHA}'] = make_netcore_pro(
            A_pro, deg_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
            variant=v, core_pro=core_pro, r=NETCORE_ALPHA)
    SENS[f'PROFANCY r={DADA_R}'] = make_profancy(P_pro, idx_pro, node_idx,
                                                 N, N_PRO, r=DADA_R)
    SENS[f'DADA-EC r={DADA_R}']  = make_dada_ec_pro(P_pro, idx_pro, N, N_PRO,
                                                    _pro_src, _pro_dst, r=DADA_R)

    for label in RUN_DATASETS:
        res_s = {}
        for mname, fn in SENS.items():
            t0 = time.time()
            res_s[mname] = run_loo_eval(dset_by_label[label], fn, node_idx, N,
                                        label=f'{mname}/{label}')
            print(f'  {label} | {mname:<24}: {(time.time()-t0)/60:.1f} min')
        print_results_table(res_s, f'{label} (tham so goc)',
                            method_order=list(SENS.keys()))

        print(f'\n  So sanh voi ban o tren (r={RWR_R}):')
        pairs = [(f'NetCore-{v} a={NETCORE_ALPHA}', f'NetCore-{v}') for v in
                 ['core', 'diff', 'ratio']] + [(f'DADA-EC r={DADA_R}', 'DADA-EC')]
        for a_nm, b_nm in pairs:
            da, db = res_s.get(a_nm), all_res[label].get(b_nm)
            if da is None or db is None:
                continue
            print(f'    {b_nm:<16} MRR {db["mrr"].mean():.4f} (r={RWR_R})  ->  '
                  f'{da["mrr"].mean():.4f} (tham so goc)   '
                  f'chenh {da["mrr"].mean()-db["mrr"].mean():+.4f}')

print('\n' + '=' * 78); print('XONG'); print('=' * 78)


# --------DONG GOP MRR THEO NHOM BAC CUA TP (thap = phan vi 1-4, cao = phan vi 5-6)
print('\n' + '=' * 78); print('PHAN G: dong gop MRR theo nhom bac cua TP'); print('=' * 78)

from evaluation import compute_metrics

N_DEG_BIN = 6
deg_met = deg_arr[met_mask]
edges = np.unique(np.quantile(deg_met, np.linspace(0, 1, N_DEG_BIN + 1)))
def deg_bin(d): return int(np.clip(np.digitize(d, edges[1:-1], right=True), 0, len(edges) - 2))
LOW_BINS, HIGH_BINS = {0, 1, 2, 3}, {4, 5}
def group_of(b): return 'low' if b in LOW_BINS else 'high'

TABLE_METHODS = [m for m in ORDER if m != 'PROFANCY']   # gom ca NetCore-core

def collect_folds(dset):
    rows = []
    for disease, mets in dset.items():
        valid = [m for m in mets if m in idx_pro]
        if len(valid) < 3: continue
        for i, test_met in enumerate(valid):
            seeds = [m for j, m in enumerate(valid) if j != i]
            seed_idx_set = {node_idx[s] for s in seeds}
            test_idx = node_idx[test_met]
            rp = compute_metrics(METHODS['PROFANCY'](seeds), test_idx, seed_idx_set, _n=N)
            if rp is None: continue
            grp = group_of(deg_bin(deg_arr[idx_pro[test_met]]))
            for m in TABLE_METHODS:
                rm = compute_metrics(METHODS[m](seeds), test_idx, seed_idx_set, _n=N)
                if rm is None: continue
                rows.append({'disease': disease, 'method': m, 'group': grp,
                            'delta': 1.0 / rm['rank'] - 1.0 / rp['rank']})
    return pd.DataFrame(rows)

def decompose(df_fold):
    n_benh = df_fold['disease'].nunique()
    t = df_fold.groupby(['disease', 'group'])['delta'].agg(['mean', 'count']).reset_index()
    t = t.merge(df_fold.groupby('disease')['delta'].count().rename('n_tong'), on='disease')
    t['dong_gop'] = (t['count'] / t['n_tong']) * t['mean']
    low  = t.loc[t['group'] == 'low',  'dong_gop'].sum() / n_benh
    high = t.loc[t['group'] == 'high', 'dong_gop'].sum() / n_benh
    return low, high, low + high

for label in RUN_DATASETS:
    df_fold = collect_folds(dset_by_label[label])
    print(f'\n=== {label} ===')
    print(f'{"Method":<22}{"contrib_low":>13}{"contrib_high":>14}{"MRR_delta":>12}{"pct_high%":>11}')
    for m in TABLE_METHODS:
        low, high, delta = decompose(df_fold[df_fold['method'] == m])
        pct = 100 * high / delta if abs(delta) > 1e-6 else float('nan')
        print(f'{m:<22}{low:>13.5f}{high:>14.5f}{delta:>12.5f}{pct:>11.1f}')