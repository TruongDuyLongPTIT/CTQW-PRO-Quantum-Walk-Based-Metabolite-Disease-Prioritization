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
def make_rwr(P_cc, node_idx, N, r=RWR_R):
    _N = N
    _P_cc = P_cc; _node_idx = node_idx
    _r = r

    def run_rwr(seed_nodes):
        valid = [s for s in seed_nodes if s in _node_idx]
        if not valid:
            return np.zeros(_N)
        p0 = np.zeros(_N)
        for s in valid:
            p0[_node_idx[s]] = 1.0 / len(valid)
        p = p0.copy()
        for _ in range(RWR_MAXITER):
            p_new = (1 - _r) * (_P_cc.T @ p) + _r * p0
            if np.abs(p_new - p).max() < RWR_TOL:
                break
            p = p_new
        return p

    return run_rwr


def make_metaborank_lite(P_cc, node_idx, N, r=RWR_R):
    """
    MetaboRank rút gọn trên G_cc (Frainay et al., 2019) — Personalized RWR /
    Global RWR, tái dùng make_rwr làm lõi. KHÔNG atom-mapping, KHÔNG PCR
    (đồ thị vô hướng nên PPR=PCR).
    """
    _run_rwr_cc = make_rwr(P_cc, node_idx, N, r=r)

    # ---- 1 lần: global RWR (seed = mọi node của G_cc, restart đều) ----
    p_global = _run_rwr_cc(list(node_idx.keys()))
    p_global_safe = np.where(p_global > 1e-15, p_global, 1e-15)

    def run_metaborank_lite(seed_nodes):
        p = _run_rwr_cc(seed_nodes)
        return p / p_global_safe

    return run_metaborank_lite


def make_ctqw_gcc(eigvals, eigvecs, node_idx, N):
    _N = N
    _node_idx = node_idx
    _ev = eigvals
    _vecs = eigvecs.astype(complex)

    def run_ctqw_gcc(seed_nodes, t=T_FIXED, _n=_N):
        valid_idx = [_node_idx[s] for s in seed_nodes if s in _node_idx]
        if not valid_idx:
            return np.zeros(_n)
        psi0 = np.zeros(_N, dtype=complex)
        psi0[valid_idx] = 1.0 / np.sqrt(len(valid_idx))
        psi_t = _vecs @ (np.exp(-1j * _ev * t) * (_vecs.T @ psi0))
        return np.abs(psi_t) ** 2

    return run_ctqw_gcc


# TABLE 2 — PROFANCY vs CTQW-PRO (G_pro)
def make_profancy(P_pro, idx_pro, node_idx, N, N_PRO, r=RWR_R):
    _run_rwr_pro = make_rwr(P_pro, idx_pro, N_PRO, r=r)   # tái dùng lõi RWR
    _idx_pro = idx_pro; _node_idx = node_idx; _N = N

    def run_profancy(seed_nodes, _n=_N):
        p = _run_rwr_pro(seed_nodes)          # RWR trên G_pro
        scores = np.zeros(_n)
        for nd, i in _node_idx.items():
            if nd in _idx_pro:
                scores[i] = p[_idx_pro[nd]]
        return scores

    return run_profancy


def make_metaborank_lite_pro(P_pro, idx_pro, node_idx, N, N_PRO, r=RWR_R):
    _run_rwr_pro = make_rwr(P_pro, idx_pro, N_PRO, r=r)
    _idx_pro = idx_pro; _node_idx = node_idx; _N = N

    # ---- 1 lần: global RWR (seed = mọi node của G_pro, restart đều) ----
    p_global = _run_rwr_pro(list(idx_pro.keys()))
    p_global_safe = np.where(p_global > 1e-15, p_global, 1e-15)

    def run_metaborank_lite_pro(seed_nodes, _n=_N):
        p = _run_rwr_pro(seed_nodes)
        ratio = p / p_global_safe
        scores = np.zeros(_n)
        for nd, i in _node_idx.items():
            if nd in _idx_pro:
                scores[i] = ratio[_idx_pro[nd]]
        return scores

    return run_metaborank_lite_pro


def make_ctqw_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst):
    _run_ctqw_gpro = make_ctqw_gcc(eigvals, eigvecs, idx_pro, N_PRO)  # tái dùng lõi CTQW, chạy trên G_pro
    _src = _pro_src; _dst = _pro_dst; _N = N

    def run_ctqw_pro(seed_nodes, t=T_FIXED, _n=_N):
        probs = _run_ctqw_gpro(seed_nodes, t=t)   # CTQW trên G_pro, kích thước N_PRO
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
        if nd in cm_set:
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
