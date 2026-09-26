import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from config import (CACHE_DIR, RESULTS_DIR, T_CTQW, NH_GAMMA, RWR_R,
                    RECON3D_CURRENCY_METABOLITE)
from graph import (parse_recon3d, build_gcc, build_gpro, compute_eigendecomp,
                   compute_coreness,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set,
                       build_eval_set1, build_eval_set3)
from methods import (make_profancy, make_metaborank_lite, make_ctqw_pro,
                     make_nh_pro, make_netcore_pro, make_dada_ec_pro,
                     make_amend_in_lite)
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

print('\n' + '=' * 78); print('PHAN A: coreness'); print('=' * 78)
t0 = time.time()
core_pro = compute_coreness(G_pro, pro_nodes)
print(f'  compute_coreness: {time.time()-t0:.2f}s')

met_mask = np.array([(nd in node_idx) for nd in pro_nodes])
cm_in_graph = [nd for nd in RECON3D_CURRENCY_METABOLITE
               if nd in idx_pro and nd in node_idx]
cm_mask = np.zeros(N_PRO, dtype=bool)
for nd in cm_in_graph:
    cm_mask[idx_pro[nd]] = True
other_mask = met_mask & (~cm_mask)

print(f'  {len(cm_in_graph)}/{len(RECON3D_CURRENCY_METABOLITE)} currency metabolite trong G_pro')
rows_A = []
for name, m in [('Currency metabolite', cm_mask), ('Metabolite thuong', other_mask)]:
    print(f'  {name:<24} n={m.sum():>5}  degree TB = {deg_pro[m].mean():.1f}'
          f'  coreness TB = {core_pro[m].mean():.1f}')
    rows_A.append({'nhom': name, 'n': int(m.sum()),
                   'degree_TB': float(deg_pro[m].mean()),
                   'coreness_TB': float(core_pro[m].mean())})
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
METHODS['MetaboRank-LITE']     = make_metaborank_lite(P_pro, idx_pro, node_idx,
                                                      N, N_PRO)
METHODS['AMEND-IN-LITE']       = make_amend_in_lite(A_pro, idx_pro, N, N_PRO,
                                                    _pro_src, _pro_dst)
METHODS['CTQW-PRO']            = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro,
                                               N, N_PRO, _pro_src, _pro_dst)
print(f'  Dung NH-CTQW-PRO (gamma={NH_GAMMA})...', end=' ', flush=True)
t0 = time.time()
METHODS['NH-CTQW-PRO']         = make_nh_pro(A_pro, idx_pro, N, N_PRO,
                                             _pro_src, _pro_dst,
                                             RECON3D_CURRENCY_METABOLITE, pro_nodes,
                                             NH_GAMMA, T_CTQW)
print(f'{time.time()-t0:.1f}s')

ORDER     = list(METHODS.keys())              # thu tu hien thi da chot
OURS      = ['CTQW-PRO', 'NH-CTQW-PRO']
CLASSICAL = [m for m in ORDER if m not in OURS]
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

# -------------KIEM DINH THONG KE------------
print('\n' + '=' * 78); print('PHAN D: kiem dinh thong ke'); print('=' * 78)


for label in RUN_DATASETS:
    print(f'\n{"#"*78}\n# {label}: tung phuong phap vs PROFANCY\n{"#"*78}')
    for m in ORDER:
        if m == 'PROFANCY':
            continue
        wilcoxon_table(all_res[label][m], all_res[label]['PROFANCY'],
                      f'{label}: {m} vs PROFANCY', method_a=m, method_b='PROFANCY')

    print(f'\n{"#"*78}\n# {label}: CTQW-PRO / NH-CTQW-PRO vs TUNG baseline\n{"#"*78}')
    for ours in OURS:
        for m in CLASSICAL:
            wilcoxon_table(all_res[label][ours], all_res[label][m],
                          f'{label}: {ours} vs {m}', method_a=ours, method_b=m)

# ------------ BANG TONG HOP CHO MUC 4.6----------
print('\n' + '=' * 78); print('PHAN E: bang tong hop (dang dua vao Muc 4.6)'); print('=' * 78)
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

