"""
03_negative_results.py — Negative results cho paper CTQW-PRO.

4 experiments theo thứ tự priority:
  EXP 1 — Self-loop leakage      ← methodological contribution
  EXP 2 — Dispersion analysis    ← when CTQW-PRO works best
  EXP 3 — Dephasing walk         ← coherence is essential
  EXP 4 — Chiral walk            ← graph not sufficiently directed
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

deg_pro_safe = np.where(deg_pro > 0, deg_pro, 1.0)
P_pro        = A_pro / deg_pro_safe[:, None]
run_profancy = make_profancy(P_pro, idx_pro, node_idx, N, N_PRO)
run_ctqw     = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO,
                              _pro_src, _pro_dst)
ctqw_fn      = lambda seeds: run_ctqw(seeds, [T_FIXED])[T_FIXED]

_ph0 = np.exp(-1j * Apro_eigvals * T_FIXED)   # reused across EXPs

def _ctqw_cpu(sidx):
    """Bare CTQW evolution: returns probs (N_PRO,)."""
    if not sidx: return np.zeros(N_PRO)
    psi0       = np.zeros(N_PRO, dtype=complex)
    psi0[sidx] = 1.0 / np.sqrt(len(sidx))
    return np.abs(Apro_eigvecs @ (_ph0 * (Apro_eigvecs.conj().T @ psi0)))**2


# ═══════════════════════════════════════════════════════════════
# EXP 1 — Self-loop leakage analysis
# (A) Baseline      — CTQW-PRO, γ=0
# (B) Leaked        — H = A_pro + γ·diag(ALL mets)
# (C) Leakage-free  — H = A_pro + γ·diag(seeds_only), per fold
# ═══════════════════════════════════════════════════════════════

import numpy as np
import torch
import time
from tqdm import tqdm

# ── Device setup ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  Device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

GAMMA = 10.0
T     = float(T_FIXED)

# ── Pre-load fixed data onto GPU ──────────────────────────────────
# A_pro: numpy float64 symmetric → GPU float64
A_pro_gpu = torch.tensor(A_pro, dtype=torch.float64, device=device)

# Condition A: reuse eigensystem already computed in setup (Apro_eigvals/vecs)
# Must cast float64 → complex128 before multiplying with 1j
eigvecs_A_c = torch.tensor(
    Apro_eigvecs, dtype=torch.float64, device=device
).to(torch.complex128)                                    # (N_PRO, N_PRO)

ph_A = torch.exp(
    -1j * torch.tensor(
        Apro_eigvals, dtype=torch.float64, device=device
    ).to(torch.complex128) * T
)                                                         # (N_PRO,) complex128

# Index arrays for G_pro → G_cc score mapping
_src_t = torch.tensor(_pro_src, dtype=torch.long, device=device)  # (E,)
_dst_t = torch.tensor(_pro_dst, dtype=torch.long, device=device)  # (E,)


# ── Helpers ───────────────────────────────────────────────────────
def make_psi0(sidx_list):
    """
    Build uniform superposition |ψ0⟩ on GPU.
    sidx_list: Python list of G_pro integer indices (no duplicates expected).
    """
    psi0 = torch.zeros(N_PRO, dtype=torch.complex128, device=device)
    if sidx_list:
        idx = torch.tensor(sidx_list, dtype=torch.long, device=device)
        psi0[idx] = 1.0 / (len(sidx_list) ** 0.5)
    return psi0


def ctqw_gpu(psi0, eigvecs_c, ph):
    """
    CTQW: |ψ(t)⟩ = V (ph ⊙ (V† |ψ0⟩)), return |ψ(t)|² as float GPU tensor.
    All args must be complex128 GPU tensors.
    """
    coeff = eigvecs_c.conj().T @ psi0   # (N_PRO,)
    psi_t = eigvecs_c @ (ph * coeff)    # (N_PRO,)
    return psi_t.abs().pow(2)           # (N_PRO,) float64


def to_cc_scores(probs_gpu):
    """
    Map G_pro probs (N_PRO,) → G_cc scores (N,), return CPU numpy array.
    Indices _src_t/_dst_t are preloaded global GPU tensors.
    """
    sc = torch.zeros(N, dtype=torch.float64, device=device)
    sc[_dst_t] = probs_gpu[_src_t]
    return sc.cpu().numpy()


def eigh_gpu(diag_idx_list):
    """
    Eigendecomp of (A_pro + GAMMA * diag(diag_idx_list)) on GPU.
    A_pro must remain symmetric after perturbation → only diagonal changes.
    Returns (eigvecs_c, ph): complex128 GPU tensors.
    Intermediate float64 tensors freed immediately to minimise VRAM.
    Uses torch.linalg.eigh (symmetric) — correct since H stays symmetric.
    """
    H = A_pro_gpu.clone()
    if diag_idx_list:
        idx = torch.tensor(diag_idx_list, dtype=torch.long, device=device)
        H[idx, idx] += GAMMA          # in-place, vectorised
    ev, vecs = torch.linalg.eigh(H)  # symmetric → real eigenvalues
    del H
    ph     = torch.exp(-1j * ev.to(torch.complex128) * T)
    vecs_c = vecs.to(torch.complex128)
    del ev, vecs                      # free float64 copies
    return vecs_c, ph


def _row_mean(res_list):
    """Average a list of metric dicts into one dict."""
    return {k: float(np.mean([r[k] for r in res_list]))
            for k in ['mrr', 'auc', 'r@20']}


def _agg(rows, k):
    """Grand mean of metric k across all disease rows."""
    vals = [r[k] for r in rows if r]
    return float(np.mean(vals)) if vals else float('nan')


# ── EXP 1 main loop ──────────────────────────────────────────────
print('\n' + '='*60)
print('EXP 1: Self-loop leakage analysis (GPU-accelerated)')
print(f'  (A) Baseline:     no self-loops (γ=0)')
print(f'  (B) Leaked:       H = A_pro + {GAMMA}·diag(ALL disease mets in G_pro)')
print(f'  (C) Leakage-free: H = A_pro + {GAMMA}·diag(seed mets only, per fold)')
print(f'  Semantics of B: test_met receives self-loop → score artificially boosted')
print(f'  Dataset: full SMPDB ({len(eval_set3)} diseases)')
print(f'  NOTE: uses torch.linalg.eigh (symmetric H) for all 3 conditions')

rows_baseline = []
rows_leaked   = []
rows_noleak   = []

t_start      = time.time()
disease_list = list(eval_set3.items())

for d_idx, (disease, mets) in enumerate(tqdm(
        disease_list, desc='EXP1', unit='disease', ncols=80)):

    valid   = [m for m in mets if m in node_idx]
    if len(valid) < 3:
        continue
    all_pro = [idx_pro[m] for m in valid if m in idx_pro]
    if not all_pro:
        continue

    # Condition B eigendecomp: one per disease (H fixed across all folds)
    # all_pro includes the test_met for every fold → that is the "leaked" part
    vecs_B_c, ph_B = eigh_gpu(all_pro)

    res_base   = []
    res_leak   = []
    res_noleak = []
    t_c_folds  = 0.0

    for i, test_met in enumerate(valid):
        seeds       = [m for j, m in enumerate(valid) if j != i]
        sidx        = [idx_pro[s] for s in seeds if s in idx_pro]
        if not sidx:
            continue
        seed_set_cc = {node_idx[s] for s in seeds if s in node_idx}
        tidx_cc     = node_idx[test_met]
        psi0        = make_psi0(sidx)

        # (A) Baseline
        sc_a = to_cc_scores(ctqw_gpu(psi0, eigvecs_A_c, ph_A))
        m_a  = compute_metrics(sc_a, tidx_cc, seed_set_cc, _n=N)
        if m_a:
            res_base.append(m_a)

        # (B) Leaked: test_met has self-loop in H_B
        sc_b = to_cc_scores(ctqw_gpu(psi0, vecs_B_c, ph_B))
        m_b  = compute_metrics(sc_b, tidx_cc, seed_set_cc, _n=N)
        if m_b:
            res_leak.append(m_b)

        # (C) Leakage-free: only seed mets get self-loop, test_met excluded
        t0 = time.time()
        vecs_C_c, ph_C = eigh_gpu(sidx)
        sc_c = to_cc_scores(ctqw_gpu(psi0, vecs_C_c, ph_C))
        del vecs_C_c, ph_C
        t_c_folds += time.time() - t0
        m_c = compute_metrics(sc_c, tidx_cc, seed_set_cc, _n=N)
        if m_c:
            res_noleak.append(m_c)

    del vecs_B_c, ph_B  # free before next disease

    if res_base:   rows_baseline.append(_row_mean(res_base))
    if res_leak:   rows_leaked.append(  _row_mean(res_leak))
    if res_noleak: rows_noleak.append(  _row_mean(res_noleak))

    # Progress log every 10 diseases
    if (d_idx + 1) % 10 == 0 or d_idx == len(disease_list) - 1:
        elapsed = time.time() - t_start
        eta     = elapsed / (d_idx + 1) * (len(disease_list) - d_idx - 1)
        mrr_a   = np.nanmean([r['mrr'] for r in rows_baseline]) if rows_baseline else 0.0
        mrr_b   = np.nanmean([r['mrr'] for r in rows_leaked])   if rows_leaked   else 0.0
        mrr_c   = np.nanmean([r['mrr'] for r in rows_noleak])   if rows_noleak   else 0.0
        tqdm.write(
            f'  [{d_idx+1:3d}/{len(disease_list)}] '
            f'elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m | '
            f'MRR  A={mrr_a:.4f}  B={mrr_b:.4f}  C={mrr_c:.4f} | '
            f'C_eigh={t_c_folds:.1f}s'
        )

# ── Final results ─────────────────────────────────────────────────
n_d     = len(rows_baseline)
total_t = time.time() - t_start

print(f'\n  Done: {n_d} diseases | total = {total_t/60:.1f} min')
print(f'\n  {"Condition":<32} {"AUC":>8} {"MRR":>8} {"R@20":>8}')
print('  ' + '-'*60)
for label, rows, note in [
    ('(A) Baseline (γ=0)',          rows_baseline, ''),
    ('(B) Self-loop LEAKED',        rows_leaked,   '  ← inflated'),
    ('(C) Self-loop leakage-free',  rows_noleak,   ''),
]:
    print(f'  {label:<32} '
          f'{_agg(rows,"auc"):>8.4f} '
          f'{_agg(rows,"mrr"):>8.4f} '
          f'{_agg(rows,"r@20"):>8.4f}{note}')

delta_leak   = _agg(rows_leaked,  'mrr') - _agg(rows_baseline, 'mrr')
delta_noleak = _agg(rows_noleak,  'mrr') - _agg(rows_baseline, 'mrr')
print(f'\n  ΔMRR (B vs A): {delta_leak:+.4f}  ← leakage artifact')
print(f'  ΔMRR (C vs A): {delta_noleak:+.4f}  ← true self-loop effect')
if delta_leak > delta_noleak + 0.01:
    print(f'  → Leakage confirmed: B inflates by '
          f'{delta_leak - delta_noleak:.4f} MRR over C')


# ═══════════════════════════════════════════════════════════════
# EXP 2 — Dispersion analysis
# Hypothesis: CTQW-PRO tốt hơn khi seeds phân tán
# Kết quả: hypothesis bị reject — CTQW-PRO tốt hơn khi CONCENTRATED
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 2: Dispersion analysis (Jaccard dissimilarity)')
print('  Hypothesis: CTQW-PRO improves MORE when seeds are dispersed')

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
        if not np.isnan(jd): jd_map[disease] = jd
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
    near_set, far_set, _ = split_by_dispersion(eval_set, set_name)
    if len(near_set) < 5 or len(far_set) < 5:
        print(f'  {set_name}: không đủ diseases — skip')
        continue
    for group_name, group_set in [('CONCENTRATED', near_set), ('DISPERSED', far_set)]:
        df_p = run_loo_eval(group_set, run_profancy, node_idx, N,
                            label=f'PROFANCY/{set_name}/{group_name}')
        df_c = run_loo_eval(group_set, ctqw_fn, node_idx, N,
                            label=f'CTQW-PRO/{set_name}/{group_name}')
        if df_p is None or df_c is None: continue
        shared    = sorted(set(df_p['disease']) & set(df_c['disease']))
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
    conc     = [r for r in disp_rows if r['group'] == 'CONCENTRATED']
    disp     = [r for r in disp_rows if r['group'] == 'DISPERSED']
    conc_map = {r['source']: r for r in conc}
    disp_map = {r['source']: r for r in disp}
    common   = set(conc_map) & set(disp_map)
    print(f'\n  KEY FINDING:')
    for r in conc: print(f'  CONCENTRATED ({r["source"]}): ΔMRR={r["delta_mrr"]:+.4f}')
    for r in disp: print(f'  DISPERSED    ({r["source"]}): ΔMRR={r["delta_mrr"]:+.4f}')
    if common and all(conc_map[s]['delta_mrr'] > disp_map[s]['delta_mrr'] for s in common):
        print('  → Hypothesis REJECTED: CTQW-PRO improves MORE when seeds are CONCENTRATED')
    elif not common:
        print('  → Insufficient data across datasets to conclude')
    pd.DataFrame(disp_rows).to_csv(RESULTS_DIR/'dispersion_analysis.csv', index=False)
    print('  Saved: dispersion_analysis.csv')


# ═══════════════════════════════════════════════════════════════
# EXP 3 — Dephasing walk
# Protocol: grid search p trên HMDB+CTD → report p* trên SMPDB
#   p=0 → CTQW thuần; p=1 → full dephasing (classical)
#   ψ_new = (1-p)*ψ + p*|ψ|  (partial phase destruction)
# ═══════════════════════════════════════════════════════════════

"""
setup_only.py — Chỉ setup dữ liệu + eigen-decomposition, KHÔNG chạy
run_loo_eval (phần tốn hàng giờ). Dùng để chạy nhanh các đoạn check
(ví dụ check_sigma5_direct.py) mà không cần chạy lại toàn bộ pipeline.

Sau khi chạy file này, các biến sau sẽ có sẵn trong kernel:
  eval_set1, idx_pro, N_PRO, N, Apro_eigvecs, Apro_eigvals,
  _ph_fixed, RNG, N_MC
"""
import numpy as np

from config import RESULTS_DIR, CACHE_DIR, T_FIXED
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon,
                   compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_cofactors_set,
                       build_eval_set1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

print('Setup (chỉ dữ liệu + eigen-decomposition, không chạy eval)...')

recon_data = parse_recon3d()
G_cc, _, N, node_idx, _, _ = build_gcc(recon_data)
pathway_mets = recon_data['pathway_mets']
met_info     = recon_data['met_info']

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
print(f'  HMDB+CTD: {len(eval_set1)} diseases')

_ph_fixed = np.exp(-1j * Apro_eigvals * float(T_FIXED))
N_MC      = 25   # đồng bộ với exp_coherence_minimal.py, dùng nếu cần

print('Setup xong. Có sẵn: eval_set1, idx_pro, N_PRO, N, Apro_eigvecs,')
print('Apro_eigvals, _ph_fixed, RNG, N_MC')
print('-> Có thể chạy thẳng check_sigma5_direct.py ngay sau đoạn này.')

#------------------------------------------------------------------------

print('\n' + '='*60)
print('EXP 3: Dephasing walk (decoherence experiment)')

import numpy as np

_test_key    = next(iter(eval_set1)) if isinstance(eval_set1, dict) else 0
_test_record = eval_set1[_test_key]
_test_seeds  = (_test_record.get('seeds', _test_record.get('seed_nodes'))
                if isinstance(_test_record, dict)
                else getattr(_test_record, 'seeds', _test_record))
print(f'Test seed set: {_test_key!r}  (n_seeds={len(_test_seeds)})')

_sidx = [idx_pro[s] for s in _test_seeds if s in idx_pro]
_psi0 = np.zeros(N_PRO, dtype=complex)
_psi0[_sidx] = 1.0 / np.sqrt(len(_sidx))
_c = Apro_eigvecs.conj().T @ _psi0


probs_analytical = (Apro_eigvecs**2) @ (np.abs(_c)**2)

def mc_probs(sigma, n_samples):
    """Vector xác suất trung bình Monte Carlo với nhiễu pha ~ N(0, sigma^2)."""
    if sigma == 0.0:
        return np.abs(Apro_eigvecs @ (_ph_fixed * _c))**2
    probs = np.zeros(N_PRO)
    for _ in range(n_samples):
        noise = RNG.normal(0.0, sigma, size=N_PRO)
        probs += np.abs(Apro_eigvecs @ ((_ph_fixed * np.exp(1j * noise)) * _c))**2
    return probs / n_samples

SIGMA_GRID = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
K_REPEATS  = 5
N_MC       = 3000

print(f'\n{"sigma":<8}{"mean L2 dist":>16}{"std":>12}   xu hướng')
print('-'*52)
prev_mean = None
for sigma in SIGMA_GRID:
    dists = [np.linalg.norm(mc_probs(sigma, N_MC) - probs_analytical)
              for _ in range(K_REPEATS)]
    m, s = np.mean(dists), np.std(dists)
    trend = '' if prev_mean is None else ('giảm' if m < prev_mean else 'bão hòa/nhiễu')
    print(f'{sigma:<8}{m:>16.6f}{s:>12.6f}   {trend}')
    prev_mean = m


# ═══════════════════════════════════════════════════════════════
# EXP 4 — Chiral Quantum Walk
# Graph metabolic không đủ directed → chiral walk không hiệu quả
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('EXP 4: Chiral Quantum Walk')
print('  Hypothesis: directed reaction information improves walk')

def _compute_directed_adj(rxn_info, idx_pro, N_PRO):
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

D_mat     = _compute_directed_adj(rxn_info, idx_pro, N_PRO)
A_antisym = D_mat - D_mat.T
n_directed = int((np.abs(A_antisym) > 0).sum() // 2)
del D_mat
print(f'  Directed edges: {n_directed}/{G_pro.number_of_edges()} '
      f'({n_directed/G_pro.number_of_edges()*100:.1f}%)')

chiral_results = {}
for phi, phi_name in zip(PHI_LIST, PHI_NAMES):
    if phi == 0:
        ev, vecs = Apro_eigvals, Apro_eigvecs
    else:
        H_chiral = A_pro.astype(complex) + 1j * phi * A_antisym
        ev, vecs = np.linalg.eigh(H_chiral); del H_chiral
    ph = np.exp(-1j * ev * T_FIXED)

    def _make_chiral_fn(vecs_=vecs, ph_=ph):
        def fn(seeds):
            valid_idx = [idx_pro[s] for s in seeds if s in idx_pro]
            if not valid_idx: return np.zeros(N)
            psi0 = np.zeros(N_PRO, dtype=complex)
            psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
            probs = np.abs(vecs_ @ (ph_ * (vecs_.conj().T @ psi0)))**2
            sc = np.zeros(N); sc[_pro_dst] = probs[_pro_src]
            return sc
        return fn

    df = run_loo_eval(eval_set3, _make_chiral_fn(), node_idx, N,
                      label=f'Chiral φ={phi_name}')
    chiral_results[phi_name] = df

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