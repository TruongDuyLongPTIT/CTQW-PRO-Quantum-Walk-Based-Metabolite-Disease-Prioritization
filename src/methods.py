"""
methods.py — Ranking methods cho CTQW-PRO pipeline.

Theo thứ tự paper:
  Table 1:  run_rwr, make_ctqw_gcc
  Table 2:  make_profancy, make_ctqw_pro
  Table 3:  make_nh_pro
  Table 4:  make_driven_pro, make_rrf
"""
import numpy as np
from scipy.stats import rankdata as _rankdata
from config import T_FIXED, RWR_R, RWR_TOL, RWR_MAXITER, DRIVEN_N_STEPS, DRIVEN_ALPHA


# ══════════════════════════════════════════════════════════════
# TABLE 1 — RWR vs CTQW (G_cc)
# ══════════════════════════════════════════════════════════════

def run_rwr(seed_nodes, P_cc, node_idx, N, r=RWR_R):
    """RWR on G_cc. p^(t+1) = (1-r)·P^T·p^t + r·p^0  (Köhler et al., 2008)."""
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
    """
    CTQW trên G_cc: ψ(t) = e^{-iAt}ψ₀
    Returns run_ctqw_gcc(seed_nodes, node_idx, t=T_FIXED) → scores (N,).
    """
    _N = N; _ev = eigvals; _vecs = eigvecs

    def run_ctqw_gcc(seed_nodes, node_idx, t=T_FIXED, _n=_N):
        valid_idx = [node_idx[s] for s in seed_nodes if s in node_idx]
        if not valid_idx: return np.zeros(_n)
        psi0 = np.zeros(_n, dtype=complex)
        psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
        psi_t = _vecs @ (np.exp(-1j * _ev * t) * (_vecs.conj().T @ psi0))
        return np.abs(psi_t)**2

    return run_ctqw_gcc


# ══════════════════════════════════════════════════════════════
# TABLE 2 — PROFANCY vs CTQW-PRO (G_pro)
# ══════════════════════════════════════════════════════════════

def make_profancy(P_pro, idx_pro, node_idx, N, N_PRO, r=RWR_R):
    """
    PROFANCY: RWR trên G_pro.
    p^(t+1) = (1-r)·P^T·p^t + r·p^0   (Köhler et al., 2008; Shang et al., 2014)
    Returns run_profancy(seed_nodes) → scores (N,).
    """
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


def _ctqw_batch_raw(seed_indices, t_values, eigvals, eigvecs, N_PRO):
    """Batch CTQW evolution helper: ψ(t) = e^{-iAt}ψ₀."""
    if not seed_indices:
        return {t: np.zeros(N_PRO) for t in t_values}
    psi0 = np.zeros(N_PRO, dtype=complex)
    psi0[seed_indices] = 1.0 / np.sqrt(len(seed_indices))
    coef      = eigvecs.conj().T @ psi0
    t_arr     = np.asarray(t_values, dtype=float)[:, None]
    phases    = np.exp(-1j * eigvals[None, :] * t_arr)
    psi_t_all = eigvecs @ (phases * coef[None, :]).T
    return {t: np.abs(psi_t_all[:, i])**2 for i, t in enumerate(t_values)}


def make_ctqw_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst):
    """
    CTQW-PRO: CTQW trên G_pro.
    Returns run_ctqw_pro(seed_nodes, t_values=None) → {t: scores (N,)}.
    """
    _N = N; _N_PRO = N_PRO
    _idx_pro = idx_pro
    _src = _pro_src; _dst = _pro_dst
    _ev = eigvals; _vecs = eigvecs

    def run_ctqw_pro(seed_nodes, t_values=None, _n=_N):
        if t_values is None: t_values = [T_FIXED]
        valid_idx = [_idx_pro[s] for s in seed_nodes if s in _idx_pro]
        raw = _ctqw_batch_raw(valid_idx, t_values, _ev, _vecs, _N_PRO)
        out = {}
        for t, probs in raw.items():
            sc = np.zeros(_n)
            sc[_dst] = probs[_src]
            out[t] = sc
        return out

    return run_ctqw_pro


# ══════════════════════════════════════════════════════════════
# TABLE 3 — NH-CTQW-PRO
# ══════════════════════════════════════════════════════════════

def make_nh_pro(A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
                cofactors, pro_nodes, gamma, t=T_FIXED):
    """
    NH-CTQW-PRO: H_eff = A_pro - i·γ·diag(cofactor_vec)
    Imaginary decay tại cofactor nodes → suppress hub bias.
    gamma = mean_degree ≈ 22.

    Returns run_nh(seed_nodes) → scores (N,).
    """
    cof_set = set(cofactors)

    # Diagonal indexing — tránh tạo N×N matrix không cần thiết
    H_eff = A_pro.astype(complex)
    for i, nd in enumerate(pro_nodes):
        if nd in cof_set or nd.replace('_c','').replace('_m','').replace('_e','') in cof_set:
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


# ══════════════════════════════════════════════════════════════
# TABLE 4 — Driven CTQW-PRO & RRF
# ══════════════════════════════════════════════════════════════

def make_driven_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst,
                    n_steps=DRIVEN_N_STEPS, alpha=DRIVEN_ALPHA):
    """
    Driven CTQW-PRO: reinforce seed state sau mỗi bước để chống temporal drift.
      ψ^(k) = normalize((1-α)·e^{-iAt}ψ^(k-1) + α·ψ_seed)
    Returns run_driven(seed_nodes) → scores (N,).
    """
    _N = N; _N_PRO = N_PRO
    _idx_pro = idx_pro; _src = _pro_src; _dst = _pro_dst
    _ev = eigvals; _vecs = eigvecs
    _phases = np.exp(-1j * eigvals * T_FIXED)

    def run_driven(seed_nodes, _n=_N):
        valid = [_idx_pro[s] for s in seed_nodes if s in _idx_pro]
        if not valid: return np.zeros(_n)
        psi_seed = np.zeros(_N_PRO, dtype=complex)
        psi_seed[valid] = 1.0 / np.sqrt(len(valid))
        psi = psi_seed.copy()
        for _ in range(n_steps):
            walked = _vecs @ (_phases * (_vecs.conj().T @ psi))
            psi    = (1 - alpha) * walked + alpha * psi_seed
            nrm    = np.linalg.norm(psi)
            if nrm > 1e-9: psi /= nrm
        sc = np.zeros(_n)
        sc[_dst] = (np.abs(psi)**2)[_src]
        return sc

    return run_driven


def make_rrf(fn_a, fn_b, k=60):
    """
    RRF(fn_a, fn_b): score_rrf(j) = 1/(k+rank_a(j)) + 1/(k+rank_b(j))
    k=60: standard default [Cormack et al., 2009].
    """
    def run_rrf(seed_nodes):
        ra = _rankdata(-fn_a(seed_nodes), method='average')
        rb = _rankdata(-fn_b(seed_nodes), method='average')
        return 1.0 / (k + ra) + 1.0 / (k + rb)
    return run_rrf

