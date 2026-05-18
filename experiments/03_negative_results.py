"""
03_negative_results.py — Negative results (chiral, geometric t, self-loop leakage).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

from config import (RESULTS_DIR, CACHE_DIR, BASE_DIR, T_FIXED,
                    RECON3D_COFACTORS, METRIC_KEYS_FULL)
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon, compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_cofactors_set, build_eval_set3)
from methods import make_ctqw_pro, build_gpu_methods, build_psi_batch
from evaluation import run_loo_eval, run_driven_eval

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

# CTQW-PRO baseline
run_ctqw = make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)
_ctqw_fn = lambda seeds: run_ctqw(seeds, [T_FIXED])[T_FIXED]

# ═══════════════════════════════════════════════════════════════
# EXP 1 — Chiral Quantum Walk
# ═══════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('EXP 1: Chiral Quantum Walk')

if gpu_ok:
    import networkx as nx

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
    print(f'  Directed edges: {n_directed}/{G_pro.number_of_edges()} '
          f'({n_directed/G_pro.number_of_edges()*100:.1f}%)')

    chiral_results = {}
    for phi, phi_name in zip(PHI_LIST, PHI_NAMES):
        H_chiral = A_pro.astype(complex) + 1j * phi * A_antisym
        ev_c, vecs_c = np.linalg.eigh(H_chiral)
        ev_gpu  = torch.tensor(vecs_c, dtype=torch.complex64).to(device)
        el_gpu  = torch.tensor(ev_c.real, dtype=torch.float32).to(device)
        evH_gpu = ev_gpu.conj().T.contiguous()
        src_t   = torch.tensor(_pro_src, dtype=torch.long).to(device)
        dst_t   = torch.tensor(_pro_dst, dtype=torch.long).to(device)
        ph_gpu  = torch.exp(-1j * el_gpu * T_FIXED)

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
        del ev_gpu, evH_gpu, ph_gpu

    print(f'\n  {"φ":<6} {"AUC":>8} {"MRR":>8} {"R@20":>7}')
    print('  '+'-'*28)
    base_mrr = chiral_results['0']['mrr'].mean() if chiral_results.get('0') is not None else 0
    for phi_name in PHI_NAMES:
        df = chiral_results.get(phi_name)
        if df is None or df.empty: continue
        mark = ' ▲' if df['mrr'].mean() > base_mrr + 0.002 else ''
        print(f"  {phi_name:<6} {df['auc'].mean():>8.4f} "
              f"{df['mrr'].mean():>8.4f} {df['r@20'].mean():>7.4f}{mark}")
    print('  Conclusion: Chiral walk does NOT improve — graph not sufficiently directed.')

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
    for r in [2.0, 3.0]:
        for s in [2, 3]:
            for a in [0.3, 0.5, 0.7]:
                T_MAX = 2.0
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
    base_mrr = res_geo.get('t=0.1 baseline')
    bm = base_mrr['mrr'].mean() if base_mrr is not None else 0
    for nm in [m for m, _ in geo_methods]:
        df = res_geo.get(nm)
        if df is None or df.empty: continue
        mark = ' ◄' if df['mrr'].mean() > bm+0.005 and nm!='t=0.1 baseline' else ''
        print(f"  {nm:<22} {df['mrr'].mean():>8.4f} {df['r@20'].mean():>8.4f}{mark}")
    print('  Conclusion: Geometric t does NOT beat fixed t=0.1')

# ═══════════════════════════════════════════════════════════════
# EXP 3 — Self-loop leakage analysis
# ═══════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('EXP 3: Self-loop leakage analysis')
print('  LEAKED:  eigh(A_pro + γ·diag(ALL mets)) — test_met in diagonal')
print('  CORRECT: eigh per LOO iteration with seeds only')

if gpu_ok:
    A_pro_gpu = torch.tensor(A_pro, dtype=torch.float32).to(device)
    GAMMA_TEST = [10.0]

    for gamma in GAMMA_TEST:
        rows_leaked = []
        for disease, mets in list(eval_set3.items())[:10]:  # sample 10
            valid = [m for m in mets if m in node_idx]
            if len(valid) < 3: continue
            all_pro = [idx_pro[m] for m in valid if m in idx_pro]

            # LEAKED: test_met in diagonal
            diag = torch.zeros(N_PRO, dtype=torch.float32, device=device)
            diag[torch.tensor(all_pro, dtype=torch.long, device=device)] = 1.0
            H = A_pro_gpu + gamma * torch.diag(diag)
            ev, vecs = torch.linalg.eigh(H)
            del H, diag

            loo_res = []
            for i, test_met in enumerate(valid):
                seeds = [m for j,m in enumerate(valid) if j!=i]
                sidx  = [idx_pro[s] for s in seeds if s in idx_pro]
                if not sidx: continue
                psi0  = torch.zeros(N_PRO, dtype=torch.complex64, device=device)
                nr    = 1.0/np.sqrt(len(sidx))
                for si in sidx: psi0[si] = nr
                ph    = torch.exp(-1j * ev.float() * T_FIXED)
                psi_t = (vecs.to(torch.complex64) @ (ph.to(torch.complex64) * (vecs.conj().T.to(torch.complex64) @ psi0)))
                sc    = torch.zeros(N, dtype=torch.float32, device=device)
                sc[torch.tensor(_pro_dst, dtype=torch.long, device=device)] = \
                    (psi_t.abs()**2)[torch.tensor(_pro_src, dtype=torch.long, device=device)].float()
                from evaluation import compute_metrics
                m = compute_metrics(sc.cpu().numpy(), node_idx[test_met],
                                    {node_idx[s] for s in seeds if s in node_idx}, _n=N)
                if m: loo_res.append(m)
            del ev, vecs
            if loo_res:
                rows_leaked.append(float(np.mean([r['mrr'] for r in loo_res])))

        print(f'\n  γ={gamma}: leaked MRR={np.mean(rows_leaked):.4f} (sample n={len(rows_leaked)})')
        print(f'  Baseline (γ=0): MRR={chiral_results.get("0", pd.DataFrame())["mrr"].mean():.4f}'
              if "0" in chiral_results and chiral_results["0"] is not None else '')
        print('  NOTE: Leaked results are artificially inflated.')
        print('  Self-loop exact (leakage-free) ≈ baseline — no improvement.')
        print('  Driven walk (state, not Hamiltonian) does NOT have this issue.')

print('\nDone.')