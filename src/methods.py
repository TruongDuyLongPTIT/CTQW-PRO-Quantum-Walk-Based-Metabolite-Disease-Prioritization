"""
methods.py — Ranking methods cho CTQW-PRO pipeline.

Theo thứ tự paper:
  Table 1:  run_rwr, make_ctqw_gcc
  Table 2:  make_profancy, make_ctqw_pro
  Table 3:  make_nh_pro
"""
import numpy as np
from config import T_FIXED, RWR_R, RWR_TOL, RWR_MAXITER


# TABLE 1 — RWR vs CTQW (G_cc)
def run_rwr(seed_nodes, P_cc, node_idx, N, r=RWR_R):
    valid = [s for s in seed_nodes if s in node_idx]
    if not valid: return np.zeros(N)
    p0 = np.zeros(N)
    for s in valid: p0[node_idx[s]] = 1.0 / len(valid)
    p = p0.copy()
    for _ in range(RWR_MAXITER):
        p_new = (1 - r) * (P_cc.T @ p) + r * p0
        if np.abs(p_new - p).max() < RWR_TOL: break
        p = p_new
    return p


def make_ctqw_gcc(eigvals, eigvecs, N):
    _N = N; _ev = eigvals; _vecs = eigvecs; _vecs = eigvecs.astype(complex)
    def run_ctqw_gcc(seed_nodes, node_idx, t=T_FIXED, _n=_N):
        valid_idx = [node_idx[s] for s in seed_nodes if s in node_idx]
        if not valid_idx: return np.zeros(_n)
        psi0 = np.zeros(_n, dtype=complex)
        psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
        psi_t = _vecs @ (np.exp(-1j * _ev * t) * (_vecs.T @ psi0))
        return np.abs(psi_t)**2

    return run_ctqw_gcc



# TABLE 2 — PROFANCY vs CTQW-PRO (G_pro)
def make_profancy(P_pro, idx_pro, node_idx, N, N_PRO, r=RWR_R):
    _N = N; _N_PRO = N_PRO
    _P_pro = P_pro; _idx_pro = idx_pro; _node_idx = node_idx
    _r = r

    def run_profancy(seed_nodes, _n=_N):
        valid = [s for s in seed_nodes if s in _idx_pro]
        if not valid: return np.zeros(_n)
        p0 = np.zeros(_N_PRO)
        for s in valid: p0[_idx_pro[s]] = 1.0 / len(valid)
        p = p0.copy()
        for _ in range(RWR_MAXITER):
            p_new = (1 - _r) * (_P_pro.T @ p) + _r * p0
            if np.abs(p_new - p).max() < RWR_TOL: break
            p = p_new
        scores = np.zeros(_n)
        for nd, i in _node_idx.items():
            if nd in _idx_pro: scores[i] = p[_idx_pro[nd]]
        return scores

    return run_profancy


def make_ctqw_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst):
    _N = N; _N_PRO = N_PRO
    _idx_pro = idx_pro
    _src = _pro_src; _dst = _pro_dst
    _ev = eigvals
    _vecs = eigvecs.astype(complex)   # ép kiểu sang complex

    def run_ctqw_pro(seed_nodes, t=T_FIXED, _n=_N):
        valid_idx = [_idx_pro[s] for s in seed_nodes if s in _idx_pro]
        if not valid_idx: return np.zeros(_n)
        psi0 = np.zeros(_N_PRO, dtype=complex)
        psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
        psi_t = _vecs @ (np.exp(-1j * _ev * t) * (_vecs.T @ psi0)) # bỏ cái .conj() đi vì lấy liên hợp phức của a + 0j thành a - 0j thì vẫn vậy, nên không cần thiết
        probs = np.abs(psi_t) ** 2
        sc = np.zeros(_n)
        sc[_dst] = probs[_src]
        return sc

    return run_ctqw_pro


# ══════════════════════════════════════════════════════════════
# TABLE 3 — NH-CTQW-PRO
# ══════════════════════════════════════════════════════════════

def make_nh_pro(A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
                CURRENCY_METABOLITE, pro_nodes, gamma, t=T_FIXED):
    cm_set = set(CURRENCY_METABOLITE)

    # Diagonal indexing — tránh tạo N×N matrix không cần thiết
    H_eff = A_pro.astype(complex)
    for i, nd in enumerate(pro_nodes):
        if nd in cm_set or nd.replace('_c','').replace('_m','').replace('_e','') in cm_set:
            H_eff[i, i] -= 1j * gamma

    eigvals_nh, V_nh = np.linalg.eig(H_eff)
    V_inv_nh  = np.linalg.inv(V_nh)
    phases_nh = np.exp(-1j * eigvals_nh * t)

    _N = N; _idx_pro = idx_pro; _src = _pro_src; _dst = _pro_dst

    def run_nh(seed_nodes, _n=_N):
        valid = [_idx_pro[s] for s in seed_nodes if s in _idx_pro]
        if not valid: return np.zeros(_n)
        psi0 = np.zeros(N_PRO, dtype=complex)
        psi0[valid] = 1.0 / np.sqrt(len(valid))
        probs = np.abs(V_nh @ (phases_nh * (V_inv_nh @ psi0)))**2
        sc = np.zeros(_n)
        sc[_dst] = probs[_src]
        return sc

    return run_nh
