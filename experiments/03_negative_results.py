"""
03_negative_results.py — Negative results cho paper CTQW-PRO.

4 experiments theo thứ tự priority:
  EXP 1 — Self-loop leakage      ← methodological contribution
  EXP 2 — Dispersion analysis    ← when CTQW-PRO works best
  EXP 3 — Dephasing walk         ← coherence is essential
  EXP 4 — Chiral walk            ← graph not sufficiently directed

Chạy trên Colab: path đã được set bởi Cell 0.
Local: sys.path.insert(0, "<project>/src")
"""
import sys
from pathlib import Path
_src = Path(__file__).resolve().parent.parent / 'src'
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
from collections import defaultdict

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED)
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon,
                   compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_cofactors_set,
                       build_eval_set1, build_eval_set3)
from methods import make_ctqw_pro, make_profancy, build_psi_batch
from evaluation import run_loo_eval, run_driven_eval, wilcoxon_table, compute_metrics

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SETUP — dùng chung cho tất cả experiments
# ═══════════════════════════════════════════════════════════════
print('='*60)
print('Setup...')
recon_data   = parse_recon3d()
G_cc, _, N, node_idx, _, _ = build_gcc(recon_data)
met_info     = recon_data['met_info']
pathway_mets = recon_data['pathway_mets']
rxn_info     = recon_data['rxn_info']

(G_pro, _, N_PRO, idx_pro,
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
COFACTORS = build_cofactors_set(hmdb_metabolites)

eval_set1, _ = build_eval_set1(
    hmdb_metabolites, hmdb_lookups, hmdb_to_recon, node_idx, COFACTORS)
eval_set3    = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx)

print(f'  HMDB+CTD: {len(eval_set1)} diseases')
print(f'  SMPDB:    {len(eval_set3)} diseases')

# GPU
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_ok = True
    print(f'  Device: {device}')
except ImportError:
    gpu_ok = False
    print('  No GPU — CPU fallback')

# Methods shared
deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro        = A_pro / deg_pro_safe[:, None]
run_profancy = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
run_ctqw     = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO,
                              _pro_src, _pro_dst)
ctqw_fn      = lambda seeds: run_ctqw(seeds, [T_FIXED])[T_FIXED]

# ═══════════════════════════════════════════════════════════════
# EXP 1 — Self-loop leakage analysis
# ← Quan trọng nhất: methodological contribution
#
# Chứng minh: self-loop encoding trong Hamiltonian (như PPI paper)
# gây data leakage trong LOO protocol.
#
# 3 conditions:
#   (A) Baseline      — CTQW-PRO, γ=0, no self-loop
#   (B) Leaked        — eigh(A_pro + γ·diag(ALL mets)) — test_met in diagonal
#   (C) Leakage-free  — eigh(A_pro + γ·diag(seeds_only)) — per LOO fold
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 1: Self-loop leakage analysis')
print('  (A) Baseline:     CTQW-PRO, γ=0, no self-loop')
print('  (B) Leaked:       eigh(A_pro + γ·diag(ALL mets)) — test_met in diagonal')
print('  (C) Leakage-free: eigh(A_pro + γ·diag(seeds_only)) — per LOO fold')

GAMMA    = 10.0  # γ >> typical edge weight (~1) để tín hiệu seed đủ mạnh
N_SAMPLE = 10
# Stratified sample theo kích thước disease
_all_sorted  = sorted(eval_set3.items(), key=lambda x: len(x[1]))
_bucket_size = max(1, len(_all_sorted) // N_SAMPLE)
sample_diseases = [_all_sorted[i * _bucket_size]
                   for i in range(min(N_SAMPLE, len(_all_sorted)))]

rows_baseline = []; rows_leaked = []; rows_noleak = []

if gpu_ok:
    A_pro_gpu = torch.tensor(A_pro, dtype=torch.float32).to(device)
    pro_src_t = torch.tensor(_pro_src, dtype=torch.long).to(device)
    pro_dst_t = torch.tensor(_pro_dst, dtype=torch.long).to(device)

    vecs0_gpu  = torch.tensor(Apro_eigvecs, dtype=torch.complex64).to(device)
    el0_gpu    = torch.tensor(Apro_eigvals, dtype=torch.float32).to(device)
    vecs0H_gpu = vecs0_gpu.conj().T.contiguous()
    ph0_gpu    = torch.exp(
        torch.tensor(-1j, dtype=torch.complex64, device=device)
        * el0_gpu.to(torch.complex64) * T_FIXED)

    def _run_ctqw_gpu(sidx, ev, evH, ph):
        if not sidx: return np.zeros(N)
        psi0 = torch.zeros(N_PRO, dtype=torch.complex64, device=device)
        psi0[sidx] = 1.0 / np.sqrt(len(sidx))
        psi_t = ev @ (ph * (evH @ psi0))
        sc    = torch.zeros(N, dtype=torch.float32, device=device)
        sc[pro_dst_t] = (psi_t.abs()**2)[pro_src_t].float()
        return sc.cpu().numpy()

    for disease, mets in sample_diseases:
        valid   = [m for m in mets if m in node_idx]
        if len(valid) < 3: continue
        all_pro = [idx_pro[m] for m in valid if m in idx_pro]
        if not all_pro: continue

        # (B) LEAKED: eigendecomp với ALL mets
        all_pro_t = torch.tensor(all_pro, dtype=torch.long, device=device)
        H_leak    = A_pro_gpu.clone()
        H_leak[all_pro_t, all_pro_t] += GAMMA
        ev_l, vecs_l = torch.linalg.eigh(H_leak)
        del H_leak, all_pro_t
        vecs_l  = vecs_l.to(torch.complex64)
        vecsH_l = vecs_l.conj().T.contiguous()
        ph_l    = torch.exp(
            torch.tensor(-1j, dtype=torch.complex64, device=device)
            * ev_l.to(torch.complex64) * T_FIXED)

        res_base = []; res_leak = []; res_noleak = []
        for i, test_met in enumerate(valid):
            seeds = [m for j, m in enumerate(valid) if j != i]
            sidx  = [idx_pro[s] for s in seeds if s in idx_pro]
            if not sidx: continue
            seed_set_cc = {node_idx[s] for s in seeds if s in node_idx}
            tidx_cc     = node_idx[test_met]

            # (A) Baseline
            sc_a = _run_ctqw_gpu(sidx, vecs0_gpu, vecs0H_gpu, ph0_gpu)
            m_a  = compute_metrics(sc_a, tidx_cc, seed_set_cc, _n=N)
            if m_a: res_base.append(m_a)

            # (B) Leaked
            sc_b = _run_ctqw_gpu(sidx, vecs_l, vecsH_l, ph_l)
            m_b  = compute_metrics(sc_b, tidx_cc, seed_set_cc, _n=N)
            if m_b: res_leak.append(m_b)

            # (C) Leakage-free — eigendecomp per fold với seeds only
            sidx_t    = torch.tensor(sidx, dtype=torch.long, device=device)
            H_correct = A_pro_gpu.clone()
            H_correct[sidx_t, sidx_t] += GAMMA
            ev_c, vecs_c = torch.linalg.eigh(H_correct)
            del H_correct, sidx_t
            vecs_c  = vecs_c.to(torch.complex64)
            vecsH_c = vecs_c.conj().T.contiguous()
            ph_c    = torch.exp(
                torch.tensor(-1j, dtype=torch.complex64, device=device)
                * ev_c.to(torch.complex64) * T_FIXED)
            sc_c = _run_ctqw_gpu(sidx, vecs_c, vecsH_c, ph_c)
            m_c  = compute_metrics(sc_c, tidx_cc, seed_set_cc, _n=N)
            del ev_c, vecs_c, vecsH_c, ph_c
            if m_c: res_noleak.append(m_c)

        del ev_l, vecs_l, vecsH_l, ph_l
        if res_base:   rows_baseline.append({k: float(np.mean([r[k] for r in res_base]))   for k in ['mrr','auc','r@20']})
        if res_leak:   rows_leaked.append(  {k: float(np.mean([r[k] for r in res_leak]))   for k in ['mrr','auc','r@20']})
        if res_noleak: rows_noleak.append(  {k: float(np.mean([r[k] for r in res_noleak])) for k in ['mrr','auc','r@20']})

    def _agg(rows, met):
        return np.mean([r[met] for r in rows]) if rows else float('nan')

    n = len(rows_baseline)
    print(f'\n  Sample: {n} diseases, γ={GAMMA}')
    print(f'\n  {"Condition":<30} {"AUC":>8} {"MRR":>8} {"R@20":>8}')
    print('  ' + '-'*58)
    print(f'  {"(A) Baseline (no self-loop)":<30} '
          f'{_agg(rows_baseline,"auc"):>8.4f} '
          f'{_agg(rows_baseline,"mrr"):>8.4f} '
          f'{_agg(rows_baseline,"r@20"):>8.4f}')
    print(f'  {"(B) Self-loop LEAKED":<30} '
          f'{_agg(rows_leaked,"auc"):>8.4f} '
          f'{_agg(rows_leaked,"mrr"):>8.4f} '
          f'{_agg(rows_leaked,"r@20"):>8.4f}  ← inflated')
    print(f'  {"(C) Self-loop leakage-free":<30} '
          f'{_agg(rows_noleak,"auc"):>8.4f} '
          f'{_agg(rows_noleak,"mrr"):>8.4f} '
          f'{_agg(rows_noleak,"r@20"):>8.4f}')

    delta_leak   = _agg(rows_leaked,  'mrr') - _agg(rows_baseline, 'mrr')
    delta_noleak = _agg(rows_noleak,  'mrr') - _agg(rows_baseline, 'mrr')
    print(f'\n  ΔMRR (B vs A): {delta_leak:+.4f}  ← gap do leakage')
    print(f'  ΔMRR (C vs A): {delta_noleak:+.4f}  ← self-loop thực sự')
    if delta_leak > delta_noleak + 0.01:
        print(f'  → Leakage artifact confirmed: B inflate hơn C bởi {delta_leak-delta_noleak:.4f} MRR')
    print(f'  NOTE: Sample {n} diseases — kết luận cuối cần chạy full eval_set3.')

else:
    print('\n  [CPU mode] Chạy baseline + leaked only (leakage-free cần GPU)')
    ph0 = np.exp(-1j * Apro_eigvals * T_FIXED)
    for disease, mets in sample_diseases:
        valid   = [m for m in mets if m in node_idx]
        if len(valid) < 3: continue
        all_pro = [idx_pro[m] for m in valid if m in idx_pro]
        if not all_pro: continue

        H_leak = A_pro.copy()
        for pi in all_pro: H_leak[pi, pi] += GAMMA
        ev_l, vecs_l = np.linalg.eigh(H_leak)
        del H_leak
        vecs_l_c = vecs_l.astype(complex)
        ph_l     = np.exp(-1j * ev_l * T_FIXED)

        res_base = []; res_leak = []
        for i, test_met in enumerate(valid):
            seeds = [m for j, m in enumerate(valid) if j != i]
            sidx  = [idx_pro[s] for s in seeds if s in idx_pro]
            if not sidx: continue
            seed_set_cc = {node_idx[s] for s in seeds if s in node_idx}
            tidx_cc     = node_idx[test_met]
            psi0        = np.zeros(N_PRO, dtype=complex)
            psi0[sidx]  = 1.0 / np.sqrt(len(sidx))

            psi_t = Apro_eigvecs @ (ph0 * (Apro_eigvecs.conj().T @ psi0))
            sc_a  = np.zeros(N); sc_a[_pro_dst] = (np.abs(psi_t)**2)[_pro_src]
            m_a   = compute_metrics(sc_a, tidx_cc, seed_set_cc, _n=N)
            if m_a: res_base.append(m_a)

            psi_l = vecs_l_c @ (ph_l * (vecs_l_c.conj().T @ psi0))
            sc_b  = np.zeros(N); sc_b[_pro_dst] = (np.abs(psi_l)**2)[_pro_src]
            m_b   = compute_metrics(sc_b, tidx_cc, seed_set_cc, _n=N)
            if m_b: res_leak.append(m_b)

        del ev_l, vecs_l, vecs_l_c
        if res_base: rows_baseline.append({'mrr': np.mean([r['mrr'] for r in res_base])})
        if res_leak: rows_leaked.append(  {'mrr': np.mean([r['mrr'] for r in res_leak])})

    bm = np.mean([r['mrr'] for r in rows_baseline]) if rows_baseline else float('nan')
    lm = np.mean([r['mrr'] for r in rows_leaked])   if rows_leaked   else float('nan')
    print(f'\n  Sample: {len(rows_baseline)} diseases, γ={GAMMA}')
    print(f'  (A) Baseline:         MRR={bm:.4f}')
    print(f'  (B) Self-loop LEAKED: MRR={lm:.4f}  ← inflated')
    print(f'  ΔMRR (B vs A): {lm-bm:+.4f}  ← gap do leakage')
    print(f'  (C) Leakage-free: cần GPU')

# ═══════════════════════════════════════════════════════════════
# EXP 2 — Dispersion analysis
# Hypothesis: CTQW-PRO tốt hơn khi seeds phân tán
# Metric: Jaccard dissimilarity giữa pathway sets của seeds
# Kết quả: hypothesis bị reject — CTQW-PRO tốt hơn khi CONCENTRATED
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 2: Dispersion analysis (Jaccard dissimilarity)')
print('  Hypothesis: CTQW-PRO improves MORE when seeds are dispersed')
print('  Metric: mean (1 - Jaccard similarity) of seed pathway sets')

# node → pathways mapping
node_to_pathways = defaultdict(set)
for pw, mets_in_pw in pathway_mets.items():
    for m in mets_in_pw:
        node_to_pathways[m].add(pw)

def jaccard_dissimilarity(nodes):
    nodes   = [n for n in nodes if n in node_to_pathways]
    k       = len(nodes)
    if k < 2: return float('nan')
    pw_sets = [node_to_pathways[n] for n in nodes]
    diss    = []
    for i in range(k):
        for j in range(i + 1, k):
            union = len(pw_sets[i] | pw_sets[j])
            inter = len(pw_sets[i] & pw_sets[j])
            diss.append(1.0 - inter / union if union > 0 else 1.0)
    return float(np.mean(diss)) if diss else float('nan')

def split_by_dispersion(eval_set, set_name, p_near=25, p_far=75):
    jd_map = {}
    for disease, mets in eval_set.items():
        valid = [m for m in mets if m in node_idx]
        if len(valid) < 3: continue
        jd = jaccard_dissimilarity(valid)
        if not np.isnan(jd):
            jd_map[disease] = jd
    if not jd_map: return {}, {}, {}
    jd_vals  = np.array(list(jd_map.values()))
    thresh_l = np.percentile(jd_vals, p_near)
    thresh_h = np.percentile(jd_vals, p_far)
    near = {d: eval_set[d] for d, jd in jd_map.items() if jd <= thresh_l}
    far  = {d: eval_set[d] for d, jd in jd_map.items() if jd >  thresh_h}
    print(f'\n  [{set_name}] P{p_near}={thresh_l:.3f} → CONCENTRATED: {len(near)} | '
          f'P{p_far}={thresh_h:.3f} → DISPERSED: {len(far)}')
    return near, far, jd_map

disp_rows = []
for eval_set, set_name in [(eval_set1, 'HMDB+CTD'), (eval_set3, 'SMPDB')]:
    near_set, far_set, jd_map = split_by_dispersion(eval_set, set_name)
    if len(near_set) < 5 or len(far_set) < 5:
        print(f'  {set_name}: không đủ diseases — skip')
        continue

    for group_name, group_set in [('CONCENTRATED', near_set), ('DISPERSED', far_set)]:
        df_p = run_loo_eval(group_set, run_profancy, node_idx, N,
                            label=f'PROFANCY/{set_name}/{group_name}')
        df_c = run_loo_eval(group_set, ctqw_fn, node_idx, N,
                            label=f'CTQW-PRO/{set_name}/{group_name}')
        if df_p is None or df_c is None: continue

        shared = sorted(set(df_p['disease']) & set(df_c['disease']))
        dp = df_p.set_index('disease'); dc = df_c.set_index('disease')
        delta_mrr = float((dc.loc[shared,'mrr'] - dp.loc[shared,'mrr']).mean())
        delta_r20 = float((dc.loc[shared,'r@20'] - dp.loc[shared,'r@20']).mean())
        disp_rows.append({'source': set_name, 'group': group_name,
                          'n': len(shared), 'delta_mrr': delta_mrr,
                          'delta_r20': delta_r20,
                          'profancy_mrr': df_p['mrr'].mean(),
                          'ctqw_mrr': df_c['mrr'].mean()})

        print(f'  {set_name:<10} {group_name:<14} '
              f'PROFANCY={df_p["mrr"].mean():.4f}  '
              f'CTQW-PRO={df_c["mrr"].mean():.4f}  '
              f'ΔMRR={delta_mrr:+.4f}  (n={len(shared)})')

        wilcoxon_table(df_c, df_p, f'{set_name}/{group_name}',
                       method_a='CTQW-PRO', method_b='PROFANCY')

if disp_rows:
    conc = [r for r in disp_rows if r['group'] == 'CONCENTRATED']
    disp = [r for r in disp_rows if r['group'] == 'DISPERSED']
    print(f'\n  KEY FINDING:')
    for r in conc:
        print(f'  CONCENTRATED ({r["source"]}): ΔMRR={r["delta_mrr"]:+.4f}')
    for r in disp:
        print(f'  DISPERSED    ({r["source"]}): ΔMRR={r["delta_mrr"]:+.4f}')
    # Match theo source explicitly để tránh zip mispairing
    conc_map = {r['source']: r for r in conc}
    disp_map = {r['source']: r for r in disp}
    common   = set(conc_map) & set(disp_map)
    if common and all(conc_map[s]['delta_mrr'] > disp_map[s]['delta_mrr']
                      for s in common):
        print(f'  → Hypothesis REJECTED: CTQW-PRO improves MORE when seeds are CONCENTRATED')
        print(f'  → Insight: quantum interference exploits pathway modularity most effectively')
        print(f'     when seeds belong to the same biochemical module.')
    elif not common:
        print(f'  → Insufficient data across datasets to conclude')
    pd.DataFrame(disp_rows).to_csv(RESULTS_DIR/'dispersion_analysis.csv', index=False)
    print(f'  Saved: dispersion_analysis.csv')

# ═══════════════════════════════════════════════════════════════
# EXP 3 — Dephasing walk
# Đại diện cho toàn bộ hướng decoherence.
# Kết luận: coherence là cơ chế then chốt, decoherence làm giảm MRR.
#
# Protocol: grid search p trên HMDB+CTD → report p* trên SMPDB
#   p=0 → CTQW thuần; p=1 → full dephasing (classical)
#   ψ_new = (1-p)*ψ + p*|ψ|  (partial phase destruction)
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 3: Dephasing walk (decoherence experiment)')
print('  Hypothesis: partial decoherence improves ranking')
print('  Protocol: grid search p on HMDB+CTD → report p* on SMPDB')

DEPHASING_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
N_STEPS_DEPH   = 3

if gpu_ok:
    # ── Build dephasing methods — reuse tensors từ EXP1 ──
    eigvecs_gpu = vecs0_gpu    # alias, không tạo mới
    eigvecsH    = vecs0H_gpu
    pro_src_t2  = pro_src_t
    pro_dst_t2  = pro_dst_t
    phases_gpu  = ph0_gpu      # exp(-i*eigvals*T_FIXED), đã tính ở EXP1

    def _ctqw_step_d(psi):
        return (eigvecs_gpu @ (phases_gpu.unsqueeze(1) * (eigvecsH @ psi.T))).T

    def _psi2scores_d(psi):
        probs  = psi.abs()**2
        scores = torch.zeros(len(psi), N, dtype=torch.float32, device=device)
        scores[:, pro_dst_t2] = probs[:, pro_src_t2].float()
        return scores

    def _norm_d(psi):
        return psi / torch.norm(psi, dim=1, keepdim=True).clamp(min=1e-9)

    def _build_psi(sidx_list):
        return build_psi_batch(sidx_list, N_PRO, device)

    def _make_deph_fn(p_val):
        def fn(psi_batch):
            psi = _norm_d(psi_batch)
            for _ in range(N_STEPS_DEPH):
                psi = _ctqw_step_d(psi)
                if p_val > 0:
                    psi = (1.0 - p_val) * psi + p_val * psi.abs().to(torch.complex64)
                psi = _norm_d(psi)
            return _psi2scores_d(psi)
        return fn

    # Include p=0.0 as fair baseline (CTQW n_steps=3)
    all_p = [0.0] + DEPHASING_GRID
    deph_methods = {f'deph_p{p:.1f}': _make_deph_fn(p) for p in all_p}

    # Phase 1: grid search on HMDB+CTD
    print(f'\n  Phase 1 — Grid search on HMDB+CTD...')
    res_grid = run_driven_eval(
        eval_set1, list(deph_methods.items()),
        node_idx, idx_pro, N, N_PRO,
        _build_psi, batch_size=128, label='GridSearch/HMDB+CTD')

    # CTQW-PRO original baseline (n_steps=1)
    df_ctqw_grid = run_loo_eval(eval_set1, ctqw_fn, node_idx, N,
                                 label='CTQW-PRO/HMDB+CTD')
    ctqw_mrr = df_ctqw_grid['mrr'].mean() if df_ctqw_grid is not None else 0.0
    df_p0    = res_grid.get('deph_p0.0')
    fair_mrr = df_p0['mrr'].mean() if (df_p0 is not None and not df_p0.empty) else ctqw_mrr

    print(f'\n  {"p":<14} {"MRR":>8} {"R@20":>8} {"AUC":>8}')
    print('  ' + '-'*40)
    print(f'  {"CTQW-PRO(t=0.1)":<14} {ctqw_mrr:>8.4f}  ← original baseline')
    print(f'  {"CTQW(n_steps=3)":<14} {fair_mrr:>8.4f}  ← fair baseline')

    best_p_name = None; best_mrr_d = fair_mrr
    for p in all_p:
        p_name = f'deph_p{p:.1f}'
        df     = res_grid.get(p_name)
        if df is None or df.empty: continue
        is_p0  = (p == 0.0)
        note   = '  (fair baseline)' if is_p0 else ''
        mark   = ' ◄ BEST' if (df['mrr'].mean() > best_mrr_d and not is_p0) else ''
        print(f'  {p_name:<14} {df["mrr"].mean():>8.4f} '
              f'{df["r@20"].mean():>8.4f} {df["auc"].mean():>8.4f}{mark}{note}')
        if df['mrr'].mean() > best_mrr_d and not is_p0:
            best_mrr_d  = df['mrr'].mean()
            best_p_name = p_name

    if best_p_name is None:
        best_p_val  = 0.0
        best_p_name = 'deph_p0.0'
        print(f'\n  No improvement — best p* = 0.0')
    else:
        best_p_val = float(best_p_name.rsplit('p', 1)[1])
        print(f'\n  Best p* = {best_p_val} (ΔMRR vs fair baseline = {best_mrr_d-fair_mrr:+.4f})')

    # Phase 2: report on SMPDB
    print(f'\n  Phase 2 — Report on SMPDB (p*={best_p_val})...')
    df_prof_s3 = run_loo_eval(eval_set3, run_profancy, node_idx, N,
                               label='PROFANCY/SMPDB')
    df_ctqw_s3 = run_loo_eval(eval_set3, ctqw_fn, node_idx, N,
                               label='CTQW-PRO/SMPDB')

    best_methods = {best_p_name: deph_methods[best_p_name]}
    res_best = run_driven_eval(
        eval_set3, list(best_methods.items()),
        node_idx, idx_pro, N, N_PRO,
        _build_psi, batch_size=128, label=f'Dephasing(p*={best_p_val})/SMPDB')
    df_deph_s3 = res_best.get(best_p_name)

    print(f'\n  {"Method":<25} {"AUC":>8} {"MRR":>8} {"R@5":>7} {"R@20":>7}')
    print('  ' + '-'*55)
    for mname, df in [('PROFANCY', df_prof_s3), ('CTQW-PRO', df_ctqw_s3),
                       (f'Dephasing(p*={best_p_val})', df_deph_s3)]:
        if df is None or df.empty:
            print(f'  {mname:<25}  (no results)'); continue
        delta = ''
        if mname not in ('PROFANCY','CTQW-PRO') and df_ctqw_s3 is not None:
            d = df['mrr'].mean() - df_ctqw_s3['mrr'].mean()
            delta = f'  Δ={d:+.4f} vs CTQW-PRO'
        print(f'  {mname:<25} {df["auc"].mean():>8.4f} {df["mrr"].mean():>8.4f} '
              f'{df["r@5"].mean():>7.4f} {df["r@20"].mean():>7.4f}{delta}')

    if df_deph_s3 is not None and not df_deph_s3.empty and df_ctqw_s3 is not None:
        print(f'\n  Wilcoxon: Dephasing(p*={best_p_val}) vs CTQW-PRO (SMPDB)')
        wilcoxon_table(df_deph_s3, df_ctqw_s3, 'SMPDB',
                       method_a=f'Dephasing(p*={best_p_val})', method_b='CTQW-PRO')
        delta_mrr = df_deph_s3['mrr'].mean() - df_ctqw_s3['mrr'].mean()
        print(f'\n  CONCLUSION: ΔMRR = {delta_mrr:+.4f}')
        if delta_mrr < -0.005:
            print(f'  ✗ Dephasing DEGRADES performance — coherence is essential')
        else:
            print(f'  ~ No significant improvement from dephasing')

else:
    # CPU fallback — chạy p=0.0 và p=0.1 (đại diện) trên SMPDB
    print('\n  [CPU mode] Running p=0.0 and p=0.1 on SMPDB...')
    phases_cpu = np.exp(-1j * Apro_eigvals * T_FIXED)

    def _make_deph_cpu(p):
        def fn(seeds):
            valid_idx = [idx_pro[s] for s in seeds if s in idx_pro]
            if not valid_idx: return np.zeros(N)
            psi = np.zeros(N_PRO, dtype=complex)
            psi[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
            for _ in range(N_STEPS_DEPH):
                psi = Apro_eigvecs @ (phases_cpu * (Apro_eigvecs.conj().T @ psi))
                if p > 0:
                    psi = (1.0-p)*psi + p*np.abs(psi).astype(complex)
                nrm = np.linalg.norm(psi)
                if nrm > 1e-9: psi /= nrm
            sc = np.zeros(N); sc[_pro_dst] = (np.abs(psi)**2)[_pro_src]
            return sc
        return fn

    df_ctqw_s3 = run_loo_eval(eval_set3, ctqw_fn, node_idx, N,
                               label='CTQW-PRO/SMPDB')
    for p in [0.0, 0.1]:
        df = run_loo_eval(eval_set3, _make_deph_cpu(p), node_idx, N,
                          label=f'Dephasing(p={p})/SMPDB')
        if df is not None and df_ctqw_s3 is not None:
            delta = df['mrr'].mean() - df_ctqw_s3['mrr'].mean()
            print(f'  p={p}: MRR={df["mrr"].mean():.4f}  ΔMRR={delta:+.4f} vs CTQW-PRO')

# ═══════════════════════════════════════════════════════════════
# EXP 4 — Chiral Quantum Walk
# Graph metabolic không đủ directed → chiral walk không hiệu quả
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 4: Chiral Quantum Walk')
print('  Hypothesis: directed reaction information improves walk')

if gpu_ok:
    def compute_directed_adj(rxn_info, idx_pro, N_PRO):
        D = np.zeros((N_PRO, N_PRO))
        for rxn in rxn_info.values():
            mets = rxn['mets']
            reac = [m for m, c in mets.items() if c < 0 and m in idx_pro]
            prod = [m for m, c in mets.items() if c > 0 and m in idx_pro]
            for r in reac:
                for p in prod:
                    D[idx_pro[r], idx_pro[p]] += 1
        return D

    PHI_LIST  = [0, np.pi/8, np.pi/4, np.pi/3, np.pi/2]
    PHI_NAMES = ['0', 'π/8', 'π/4', 'π/3', 'π/2']

    D_mat      = compute_directed_adj(rxn_info, idx_pro, N_PRO)
    A_antisym  = D_mat - D_mat.T
    n_directed = int((np.abs(A_antisym) > 0).sum() // 2)
    del D_mat
    print(f'  Directed edges: {n_directed}/{G_pro.number_of_edges()} '
          f'({n_directed/G_pro.number_of_edges()*100:.1f}%)')

    # reuse pro_src_t/pro_dst_t từ EXP1
    src_t4 = pro_src_t
    dst_t4 = pro_dst_t

    chiral_results = {}
    for phi, phi_name in zip(PHI_LIST, PHI_NAMES):
        if phi == 0:
            ev_gpu  = torch.tensor(Apro_eigvecs, dtype=torch.complex64).to(device)
            el_gpu  = torch.tensor(Apro_eigvals, dtype=torch.float32).to(device)
            evH_gpu = ev_gpu.conj().T.contiguous()
        else:
            H_chiral = A_pro.astype(complex) + 1j * phi * A_antisym
            ev_c, vecs_c = np.linalg.eigh(H_chiral)
            del H_chiral
            ev_gpu  = torch.tensor(vecs_c, dtype=torch.complex64).to(device)
            el_gpu  = torch.tensor(ev_c.real, dtype=torch.float32).to(device)
            del ev_c, vecs_c
        ph_gpu = torch.exp(
            torch.tensor(-1j, dtype=torch.complex64, device=device)
            * el_gpu.to(torch.complex64) * T_FIXED)

        def _make_chiral(ev, evH, ph, src, dst):
            def fn(seeds):
                valid_idx = [idx_pro[s] for s in seeds if s in idx_pro]
                if not valid_idx: return np.zeros(N)
                psi0 = torch.zeros(N_PRO, dtype=torch.complex64, device=device)
                psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
                psi_t = ev @ (ph * (evH @ psi0))
                sc    = torch.zeros(N, dtype=torch.float32, device=device)
                sc[dst] = (psi_t.abs()**2)[src].float()
                return sc.cpu().numpy()
            return fn

        chiral_fn = _make_chiral(ev_gpu, evH_gpu, ph_gpu, src_t4, dst_t4)
        df = run_loo_eval(eval_set3, chiral_fn, node_idx, N,
                          label=f'Chiral φ={phi_name}')
        chiral_results[phi_name] = df
        del ev_gpu, evH_gpu, el_gpu, ph_gpu

    del A_antisym

    print(f'\n  {"φ":<6} {"AUC":>8} {"MRR":>8} {"R@20":>7}')
    print('  ' + '-'*28)
    base_mrr_c = (chiral_results['0']['mrr'].mean()
                  if chiral_results.get('0') is not None else 0)
    for phi_name in PHI_NAMES:
        df = chiral_results.get(phi_name)
        if df is None or df.empty: continue
        mark = ' ▲' if df['mrr'].mean() > base_mrr_c + 0.002 else ''
        print(f"  {phi_name:<6} {df['auc'].mean():>8.4f} "
              f"{df['mrr'].mean():>8.4f} {df['r@20'].mean():>7.4f}{mark}")

    best_phi = max(
        (p for p in PHI_NAMES
         if chiral_results.get(p) is not None and not chiral_results[p].empty),
        key=lambda p: chiral_results[p]['mrr'].mean(), default='0')
    best_mrr_c = (chiral_results[best_phi]['mrr'].mean()
                  if chiral_results.get(best_phi) is not None else 0)

    if best_mrr_c > base_mrr_c + 0.005 and best_phi != '0':
        print(f'  Chiral walk IMPROVED at φ={best_phi} '
              f'(ΔMRR={best_mrr_c-base_mrr_c:+.4f}) — unexpected, check.')
    else:
        print(f'  Chiral walk does NOT improve '
              f'(best φ={best_phi}, ΔMRR={best_mrr_c-base_mrr_c:+.4f}).')
        print(f'  Reason: only {n_directed}/{G_pro.number_of_edges()} '
              f'directed edges — graph not sufficiently asymmetric.')

print('\nDone.')