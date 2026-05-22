"""
03_negative_results.py — Negative results (chiral, geometric t, self-loop leakage).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED)
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon, compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_eval_set3)
from methods import make_ctqw_pro, build_gpu_methods, build_psi_batch
from evaluation import run_loo_eval, run_driven_eval, compute_metrics

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print('Setup...')
recon_data   = parse_recon3d()
G_cc, graph_nodes, N, node_idx, A_cc, degrees = build_gcc(recon_data)
met_info     = recon_data['met_info']
pathway_mets = recon_data['pathway_mets']
rxn_info     = recon_data['rxn_info']

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

Apro_eigvals, Apro_eigvecs = compute_eigendecomp(A_pro, CACHE_DIR/'gpro_eigdecomp.npz')

hmdb_data        = parse_hmdb()
hmdb_metabolites = hmdb_data['metabolites']
hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)
hmdb_to_recon    = build_hmdb_to_recon_initial(met_info, node_idx)
augment_hmdb_to_recon(hmdb_to_recon, met_info, node_idx,
    hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
    hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'])

eval_set3 = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx)
print(f'  SMPDB: {len(eval_set3)} diseases')

try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_ok = True
    print(f'  Device: {device}')
except ImportError:
    gpu_ok = False

# CTQW-PRO baseline (eigendecomp cached, used directly in EXP3)
run_ctqw = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)

# ═══════════════════════════════════════════════════════════════
# EXP 1 — Chiral Quantum Walk
# ═══════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('EXP 1: Chiral Quantum Walk')

if gpu_ok:
    def compute_directed_adj(rxn_info, idx_pro, N_PRO):
        D = np.zeros((N_PRO, N_PRO))
        for rxn in rxn_info.values():
            mets  = rxn['mets']
            reac  = [m for m, c in mets.items() if c < 0 and m in idx_pro]
            prod  = [m for m, c in mets.items() if c > 0 and m in idx_pro]
            for r in reac:
                for p in prod:
                    D[idx_pro[r], idx_pro[p]] += 1
        return D

    PHI_LIST  = [0, np.pi/8, np.pi/4, np.pi/3, np.pi/2]
    PHI_NAMES = ['0', 'π/8', 'π/4', 'π/3', 'π/2']

    D_mat     = compute_directed_adj(rxn_info, idx_pro, N_PRO)
    A_antisym = D_mat - D_mat.T
    n_directed = int((np.abs(A_antisym) > 0).sum() // 2)
    del D_mat  # 67MB, không cần nữa sau khi tính A_antisym
    print(f'  Directed edges: {n_directed}/{G_pro.number_of_edges()} '
          f'({n_directed/G_pro.number_of_edges()*100:.1f}%)')

    # src_t, dst_t không thay đổi giữa các phi — tạo 1 lần
    src_t = torch.tensor(_pro_src, dtype=torch.long).to(device)
    dst_t = torch.tensor(_pro_dst, dtype=torch.long).to(device)

    # A_antisym dùng trong loop — del sau khi loop xong
    chiral_results = {}
    for phi, phi_name in zip(PHI_LIST, PHI_NAMES):
        if phi == 0:
            # phi=0: H_chiral = A_pro — reuse existing eigdecomp, skip 134MB alloc
            ev_gpu  = torch.tensor(Apro_eigvecs, dtype=torch.complex64).to(device)
            el_gpu  = torch.tensor(Apro_eigvals, dtype=torch.float32).to(device)
            evH_gpu = ev_gpu.conj().T.contiguous()
        else:
            H_chiral = A_pro.astype(complex) + 1j * phi * A_antisym
            ev_c, vecs_c = np.linalg.eigh(H_chiral)
            del H_chiral
            ev_gpu  = torch.tensor(vecs_c, dtype=torch.complex64).to(device)
            el_gpu  = torch.tensor(ev_c.real, dtype=torch.float32).to(device)
            del ev_c, vecs_c  # numpy arrays, free immediately
        ph_gpu  = torch.exp(torch.tensor(-1j, dtype=torch.complex64, device=device) * el_gpu.to(torch.complex64) * T_FIXED)

        def _make_chiral(ev, evH, ph, src, dst):
            def fn(seeds):
                valid_idx = [idx_pro[s] for s in seeds if s in idx_pro]
                if not valid_idx: return np.zeros(N)
                psi0 = np.zeros(N_PRO, dtype=complex)
                nr   = 1.0/np.sqrt(len(valid_idx))
                for i in valid_idx: psi0[i] = nr
                psi0_t = torch.tensor(psi0, dtype=torch.complex64).to(device)
                coef   = evH @ psi0_t
                psi_t  = (ev @ (ph * coef))
                probs  = psi_t.abs()**2
                sc     = torch.zeros(N, dtype=torch.float32, device=device)
                sc[dst] = probs[src].float()
                return sc.cpu().numpy()
            return fn

        chiral_fn = _make_chiral(ev_gpu, evH_gpu, ph_gpu, src_t, dst_t)
        df = run_loo_eval(eval_set3, chiral_fn, node_idx, N, label=f'Chiral φ={phi_name}')
        chiral_results[phi_name] = df
        del ev_gpu, evH_gpu, el_gpu, ph_gpu

    del A_antisym  # 67MB, không cần nữa sau loop
    print(f'\n  {"φ":<6} {"AUC":>8} {"MRR":>8} {"R@20":>7}')
    print('  '+'-'*28)
    base_mrr = chiral_results['0']['mrr'].mean() if chiral_results.get('0') is not None else 0
    for phi_name in PHI_NAMES:
        df = chiral_results.get(phi_name)
        if df is None or df.empty: continue
        mark = ' ▲' if df['mrr'].mean() > base_mrr + 0.002 else ''
        print(f"  {phi_name:<6} {df['auc'].mean():>8.4f} "
              f"{df['mrr'].mean():>8.4f} {df['r@20'].mean():>7.4f}{mark}")
    best_phi = max((phi_name for phi_name in PHI_NAMES
                    if chiral_results.get(phi_name) is not None
                    and not chiral_results[phi_name].empty),
                   key=lambda p: chiral_results[p]['mrr'].mean(),
                   default='0')
    best_mrr = chiral_results[best_phi]['mrr'].mean() if chiral_results.get(best_phi) is not None else 0
    if best_mrr > base_mrr + 0.005 and best_phi != '0':
        print(f'  Chiral walk IMPROVED at φ={best_phi} (ΔMRR={best_mrr-base_mrr:+.4f}) — unexpected, check.')
    else:
        print(f'  Chiral walk does NOT improve (best φ={best_phi}, ΔMRR={best_mrr-base_mrr:+.4f}).')
        print(f'  Reason: only {n_directed}/{G_pro.number_of_edges()} directed edges — graph not sufficiently asymmetric.')

# ═══════════════════════════════════════════════════════════════
# EXP 2 — Geometric t schedule
# ═══════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('EXP 2: Geometric t schedule')

if gpu_ok:
    gpu_fns = build_gpu_methods(
        Apro_eigvals, Apro_eigvecs, _pro_src, _pro_dst, N, N_PRO, device=device)

    def _build_psi(sidx_list):
        return build_psi_batch(sidx_list, N_PRO, device)

    ctqw_step   = gpu_fns['_ctqw_step']
    psi2scores  = gpu_fns['_psi_to_scores']
    norm_psi    = gpu_fns['_norm_psi']

    geo_methods = [
        ('t=0.1 baseline', gpu_fns['ctqw_pro']),
        ('driven_s2_a0.5', gpu_fns['driven']),
    ]
    T_MAX = 2.0  # max time cap for geometric schedule
    for r in [2.0, 3.0]:
        for s in [2, 3]:
            for a in [0.3, 0.5, 0.7]:
                t_steps = [min(0.1*(r**k), T_MAX) for k in range(s)]
                def _make_geo(ts, al):
                    def fn(psi):
                        pn  = norm_psi(psi)
                        ps  = pn.clone()
                        for t_k in ts:
                            ps = (1-al)*ctqw_step(ps, t_k) + al*pn
                            ps = ps/torch.norm(ps, dim=1, keepdim=True).clamp(min=1e-9)
                        return psi2scores(ps)
                    return fn
                geo_methods.append((f'r{int(r)}_s{s}_a{a}', _make_geo(t_steps, a)))

    res_geo = run_driven_eval(eval_set3, geo_methods, node_idx, idx_pro, N, N_PRO,
                               _build_psi, batch_size=32, label='Geometric-t')

    print(f'\n  {"Method":<22} {"MRR":>8} {"R@20":>8}')
    print('  '+'-'*40)
    base_df_geo = res_geo.get('t=0.1 baseline')
    bm = base_df_geo['mrr'].mean() if base_df_geo is not None else 0
    for nm in [m for m, _ in geo_methods]:
        df = res_geo.get(nm)
        if df is None or df.empty: continue
        mark = ' ◄' if df['mrr'].mean() > bm+0.005 and nm!='t=0.1 baseline' else ''
        print(f"  {nm:<22} {df['mrr'].mean():>8.4f} {df['r@20'].mean():>8.4f}{mark}")
    best_geo = max((nm for nm, _ in geo_methods if nm != 't=0.1 baseline'
                    and res_geo.get(nm) is not None
                    and not res_geo[nm].empty),
                   key=lambda nm: res_geo[nm]['mrr'].mean(),
                   default=None)
    if best_geo and res_geo[best_geo]['mrr'].mean() > bm + 0.005:
        print(f'  Geometric t IMPROVED: {best_geo} (ΔMRR={res_geo[best_geo]["mrr"].mean()-bm:+.4f}) — unexpected, check.')
    else:
        best_delta = (res_geo[best_geo]['mrr'].mean() - bm) if best_geo and res_geo.get(best_geo) is not None else 0
        print(f'  Geometric t does NOT beat fixed t=0.1 (best ΔMRR={best_delta:+.4f}).')

# ═══════════════════════════════════════════════════════════════
# EXP 3 — Self-loop leakage analysis
# 3 conditions so sánh trên cùng sample 10 diseases:
#   (A) Baseline      — CTQW-PRO, không có self-loop (γ=0)
#   (B) Leaked        — self-loop trên ALL mets kể cả test_met
#                       → Hamiltonian tính 1 lần/disease (LEAKED)
#   (C) Leakage-free  — self-loop chỉ trên seeds của từng LOO fold
#                       → Hamiltonian tính lại mỗi fold (CORRECT)
# ═══════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('EXP 3: Self-loop leakage analysis')
print('  (A) Baseline:     CTQW-PRO, γ=0, no self-loop')
print('  (B) Leaked:       eigh(A_pro + γ·diag(ALL mets)) — test_met in diagonal')
print('  (C) Leakage-free: eigh(A_pro + γ·diag(seeds_only)) — per LOO fold')


GAMMA    = 10.0  # self-loop strength: γ=10 >> typical edge weight (~1) để tín hiệu seed đủ mạnh
N_SAMPLE = 10  # sample để chạy nhanh
# Stratified sample: lấy đều theo kích thước disease thay vì 10 đầu tiên
_all_diseases = list(eval_set3.items())
_all_diseases_sorted = sorted(_all_diseases, key=lambda x: len(x[1]))
# Lấy đều: chia thành N_SAMPLE bucket, lấy 1 disease/bucket
_bucket_size = max(1, len(_all_diseases_sorted) // N_SAMPLE)
sample_diseases = [_all_diseases_sorted[i * _bucket_size]
                   for i in range(min(N_SAMPLE, len(_all_diseases_sorted)))]

rows_baseline  = []   # (A)
rows_leaked    = []   # (B)
rows_noleak    = []   # (C)

if gpu_ok:
    A_pro_gpu  = torch.tensor(A_pro,  dtype=torch.float32).to(device)
    pro_src_t  = torch.tensor(_pro_src, dtype=torch.long).to(device)
    pro_dst_t  = torch.tensor(_pro_dst, dtype=torch.long).to(device)

    # Pre-compute baseline eigvecs (γ=0, dùng lại từ Apro_eigvals/vecs)
    vecs0_gpu = torch.tensor(Apro_eigvecs, dtype=torch.complex64).to(device)
    el0_gpu   = torch.tensor(Apro_eigvals, dtype=torch.float32).to(device)
    vecs0H_gpu = vecs0_gpu.conj().T.contiguous()
    ph0_gpu   = torch.exp(torch.tensor(-1j, dtype=torch.complex64, device=device) * el0_gpu.to(torch.complex64) * T_FIXED)

    def _run_ctqw_gpu(sidx, ev, evH, ph):
        """Helper: chạy CTQW cho một tập seeds, trả về scores (N,)."""
        if not sidx: return np.zeros(N)
        psi0 = torch.zeros(N_PRO, dtype=torch.complex64, device=device)
        nr   = 1.0 / np.sqrt(len(sidx))
        for si in sidx: psi0[si] = nr
        coef  = evH @ psi0
        psi_t = ev @ (ph * coef)
        probs = psi_t.abs()**2
        sc    = torch.zeros(N, dtype=torch.float32, device=device)
        sc[pro_dst_t] = probs[pro_src_t].float()
        return sc.cpu().numpy()

    for disease, mets in sample_diseases:
        valid   = [m for m in mets if m in node_idx]
        if len(valid) < 3: continue
        all_pro = [idx_pro[m] for m in valid if m in idx_pro]
        if not all_pro: continue

        # ── (B) LEAKED: eigendecomp 1 lần với ALL mets ──────────
        # Dùng diagonal index thay vì torch.diag() để tránh tạo N²×N² matrix
        all_pro_t = torch.tensor(all_pro, dtype=torch.long, device=device)
        H_leak    = A_pro_gpu.clone()
        H_leak[all_pro_t, all_pro_t] += GAMMA
        ev_l, vecs_l = torch.linalg.eigh(H_leak)
        del H_leak, all_pro_t
        vecs_l  = vecs_l.to(torch.complex64)
        vecsH_l = vecs_l.conj().T.contiguous()
        ph_l    = torch.exp(torch.tensor(-1j, dtype=torch.complex64, device=device) * ev_l.to(torch.complex64) * T_FIXED)

        res_base = []; res_leak = []; res_noleak = []

        for i, test_met in enumerate(valid):
            seeds = [m for j, m in enumerate(valid) if j != i]
            sidx  = [idx_pro[s] for s in seeds if s in idx_pro]
            if not sidx: continue
            seed_set_cc = {node_idx[s] for s in seeds if s in node_idx}
            tidx_cc     = node_idx[test_met]

            # (A) Baseline — dùng eigvecs đã tính sẵn (γ=0)
            sc_a = _run_ctqw_gpu(sidx, vecs0_gpu, vecs0H_gpu, ph0_gpu)
            m_a  = compute_metrics(sc_a, tidx_cc, seed_set_cc, _n=N)
            if m_a: res_base.append(m_a)

            # (B) Leaked — dùng eigvecs từ ALL mets diagonal
            sc_b = _run_ctqw_gpu(sidx, vecs_l, vecsH_l, ph_l)
            m_b  = compute_metrics(sc_b, tidx_cc, seed_set_cc, _n=N)
            if m_b: res_leak.append(m_b)

            # (C) Leakage-free — eigendecomp lại với seeds only
            sidx_t    = torch.tensor(sidx, dtype=torch.long, device=device)
            H_correct = A_pro_gpu.clone()
            H_correct[sidx_t, sidx_t] += GAMMA
            ev_c, vecs_c = torch.linalg.eigh(H_correct)
            del H_correct, sidx_t
            vecs_c  = vecs_c.to(torch.complex64)
            vecsH_c = vecs_c.conj().T.contiguous()
            ph_c    = torch.exp(torch.tensor(-1j, dtype=torch.complex64, device=device) * ev_c.to(torch.complex64) * T_FIXED)
            sc_c    = _run_ctqw_gpu(sidx, vecs_c, vecsH_c, ph_c)
            m_c     = compute_metrics(sc_c, tidx_cc, seed_set_cc, _n=N)
            del ev_c, vecs_c, vecsH_c, ph_c
            if m_c: res_noleak.append(m_c)

        del ev_l, vecs_l, vecsH_l, ph_l

        if res_base:   rows_baseline.append({k: float(np.mean([r[k] for r in res_base]))   for k in ['mrr','auc','r@20']})
        if res_leak:   rows_leaked.append(  {k: float(np.mean([r[k] for r in res_leak]))   for k in ['mrr','auc','r@20']})
        if res_noleak: rows_noleak.append(  {k: float(np.mean([r[k] for r in res_noleak])) for k in ['mrr','auc','r@20']})

    # ── In kết quả ──────────────────────────────────────────────
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
    print(f'\n  Nhận xét (sample={n} diseases — xem full eval để conclude):')
    print(f'  Gap B-A ({delta_leak:+.4f}) phản ánh leakage artifact.')
    print(f'  Gap C-A ({delta_noleak:+.4f}) phản ánh tín hiệu self-loop thực.')
    if delta_leak > delta_noleak + 0.01:
        print(f'  → B inflate hơn C: leakage đang inflate kết quả self-loop.')
    print(f'  Driven walk (reinforce state, không sửa Hamiltonian) không có vấn đề này.')
    print(f'  NOTE: Sample {n} diseases — kết luận cuối cần chạy full eval_set3.')

else:
    # CPU fallback — chỉ chạy baseline + leaked (leakage-free quá chậm trên CPU)
    print('\n  [CPU mode] Chỉ chạy baseline và leaked (leakage-free cần GPU)')
    ph0 = np.exp(-1j * Apro_eigvals * T_FIXED)  # không đổi giữa các diseases
    for disease, mets in sample_diseases:
        valid   = [m for m in mets if m in node_idx]
        if len(valid) < 3: continue
        all_pro = [idx_pro[m] for m in valid if m in idx_pro]
        if not all_pro: continue

        # Leaked: eigendecomp với ALL mets
        # Cộng GAMMA vào diagonal thay vì tạo full N²×N² diagonal matrix
        H_leak = A_pro.copy()
        for pi in all_pro: H_leak[pi, pi] += GAMMA
        ev_l, vecs_l = np.linalg.eigh(H_leak)
        del H_leak

        # vecs_l_c và ph_l tính 1 lần/disease, không thay đổi giữa các fold
        vecs_l_c = vecs_l.astype(complex)
        ph_l     = np.exp(-1j * ev_l * T_FIXED)

        res_base = []; res_leak = []
        for i, test_met in enumerate(valid):
            seeds = [m for j, m in enumerate(valid) if j != i]
            sidx  = [idx_pro[s] for s in seeds if s in idx_pro]
            if not sidx: continue
            seed_set_cc = {node_idx[s] for s in seeds if s in node_idx}
            tidx_cc     = node_idx[test_met]

            # Baseline
            psi0 = np.zeros(N_PRO, dtype=complex)
            nr   = 1.0 / np.sqrt(len(sidx))
            for si in sidx: psi0[si] = nr
            psi_t = Apro_eigvecs @ (ph0 * (Apro_eigvecs.conj().T @ psi0))
            sc_a  = np.zeros(N); sc_a[_pro_dst] = (np.abs(psi_t)**2)[_pro_src]
            m_a   = compute_metrics(sc_a, tidx_cc, seed_set_cc, _n=N)
            if m_a: res_base.append(m_a)

            # Leaked — dùng vecs_l_c và ph_l đã tính sẵn
            psi_l = vecs_l_c @ (ph_l * (vecs_l_c.conj().T @ psi0))
            sc_b  = np.zeros(N); sc_b[_pro_dst] = (np.abs(psi_l)**2)[_pro_src]
            m_b   = compute_metrics(sc_b, tidx_cc, seed_set_cc, _n=N)
            if m_b: res_leak.append(m_b)

        del ev_l, vecs_l, vecs_l_c
        if res_base:  rows_baseline.append({'mrr': np.mean([r['mrr'] for r in res_base])})
        if res_leak:  rows_leaked.append(  {'mrr': np.mean([r['mrr'] for r in res_leak])})

    n = len(rows_baseline)
    bm = np.mean([r['mrr'] for r in rows_baseline]) if rows_baseline else float('nan')
    lm = np.mean([r['mrr'] for r in rows_leaked])   if rows_leaked   else float('nan')
    print(f'\n  Sample: {n} diseases, γ={GAMMA}')
    print(f'  {"(A) Baseline":<30} MRR={bm:.4f}')
    print(f'  {"(B) Self-loop LEAKED":<30} MRR={lm:.4f}  ← inflated')
    print(f'  ΔMRR (B vs A): {lm-bm:+.4f}  ← gap do leakage')
    print(f'  (C) Leakage-free: cần GPU để chạy đủ nhanh')

print('\nDone.')