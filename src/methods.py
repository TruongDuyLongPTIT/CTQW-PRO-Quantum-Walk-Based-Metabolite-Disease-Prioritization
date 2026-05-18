"""
methods.py — Tất cả ranking methods.
Exact từ notebook Cell 6 (PROFANCY + CTQW-PRO) và Cell 9 (GPU Driven).

Hardening pattern: _n=N default arg captures graph size at definition time.
"""
import numpy as np
from config import T_FIXED, RWR_ALPHA, RWR_TOL, RWR_MAXITER


# ── RWR trên G_cc (Table 1 baseline) ─────────────────────────────────────────

def run_rwr(seed_nodes, P_cc, node_idx, N, alpha=RWR_ALPHA):
    """RWR on G_cc. Baseline for Table 1."""
    valid = [s for s in seed_nodes if s in node_idx]
    if not valid: return np.zeros(N)
    e0 = np.zeros(N)
    for s in valid: e0[node_idx[s]] = 1.0 / len(valid)
    r = e0.copy()
    for _ in range(RWR_MAXITER):
        r_new = alpha * (P_cc.T @ r) + (1 - alpha) * e0
        if np.abs(r_new - r).max() < RWR_TOL: break
        r = r_new
    return r


# ── PROFANCY: RWR trên G_pro — exact từ notebook Cell 6 run_profancy() ───────

def make_profancy(P_pro, idx_pro, node_idx, N, N_PRO, alpha=RWR_ALPHA):
    """
    Returns run_profancy(seed_nodes) → scores (N,).
    Factory pattern: captures all state at definition time (no globals needed).
    Mirrors notebook's hardened closure: _n=N default arg.
    """
    _N     = N
    _N_PRO = N_PRO
    _P_pro = P_pro
    _idx_pro  = idx_pro
    _node_idx = node_idx

    def run_profancy(seed_nodes, _n=_N):
        """Exact từ notebook Cell 6."""
        valid = [s for s in seed_nodes if s in _idx_pro]
        if not valid: return np.zeros(_n)

        e0 = np.zeros(_N_PRO)
        for s in valid: e0[_idx_pro[s]] = 1.0 / len(valid)

        r = e0.copy()
        for _ in range(RWR_MAXITER):
            r_new = alpha * (_P_pro.T @ r) + (1 - alpha) * e0
            if np.abs(r_new - r).max() < RWR_TOL: break
            r = r_new

        # Exact từ notebook:
        #   for nd, i in node_idx.items():
        #       if nd in idx_pro: scores[i] = r[idx_pro[nd]]
        scores = np.zeros(_n)
        for nd, i in _node_idx.items():
            if nd in _idx_pro: scores[i] = r[_idx_pro[nd]]
        return scores

    return run_profancy


# ── CTQW-PRO — exact từ notebook Cell 6 ──────────────────────────────────────

def _ctqw_batch_raw(seed_indices, t_values, Apro_eigvals, Apro_eigvecs, N_PRO):
    """
    Batch CTQW evolution.
    Exact từ notebook Cell 6 _ctqw_batch_raw().
    """
    if not seed_indices:
        return {t: np.zeros(N_PRO) for t in t_values}
    psi0 = np.zeros(N_PRO, dtype=complex)
    norm = 1.0 / np.sqrt(len(seed_indices))
    for idx in seed_indices: psi0[idx] = norm
    coef      = Apro_eigvecs.conj().T @ psi0
    t_arr     = np.asarray(t_values, dtype=float)[:, None]
    phases    = np.exp(-1j * Apro_eigvals[None, :] * t_arr)
    psi_t_all = Apro_eigvecs @ (phases * coef[None, :]).T
    return {t: np.abs(psi_t_all[:, i])**2 for i, t in enumerate(t_values)}


def make_ctqw_pro(Apro_eigvals, Apro_eigvecs, idx_pro, N, N_PRO,
                  _pro_src, _pro_dst):
    """
    Returns run_ctqw_pro(seed_nodes, t_values=None) → {t: scores}.
    Exact từ notebook Cell 6 run_ctqw_pro().
    """
    _N     = N
    _N_PRO = N_PRO
    _idx_pro  = idx_pro
    _pro_src_ = _pro_src
    _pro_dst_ = _pro_dst
    _eigvals  = Apro_eigvals
    _eigvecs  = Apro_eigvecs

    def run_ctqw_pro(seed_nodes, t_values=None, _n=_N):
        if t_values is None: t_values = [T_FIXED]
        valid_idx = [_idx_pro[s] for s in seed_nodes if s in _idx_pro]
        raw = _ctqw_batch_raw(valid_idx, t_values, _eigvals, _eigvecs, _N_PRO)
        out = {}
        for t, probs in raw.items():
            sc = np.zeros(_n)
            sc[_pro_dst_] = probs[_pro_src_]
            out[t] = sc
        return out

    return run_ctqw_pro


# ── CTQW on G_cc (Table 1) ────────────────────────────────────────────────────

def make_ctqw_gcc(Acc_eigvals, Acc_eigvecs, N):
    """
    CTQW trên G_cc (không có pathway nodes).
    Scores ở G_cc space — không cần _pro_src/_pro_dst.
    """
    _N        = N
    _eigvals  = Acc_eigvals
    _eigvecs  = Acc_eigvecs

    def run_ctqw_gcc(seed_nodes, node_idx, t=T_FIXED, _n=_N):
        valid_idx = [node_idx[s] for s in seed_nodes if s in node_idx]
        if not valid_idx: return np.zeros(_n)
        psi0 = np.zeros(_n, dtype=complex)
        norm = 1.0 / np.sqrt(len(valid_idx))
        for idx in valid_idx: psi0[idx] = norm
        coef   = _eigvecs.conj().T @ psi0
        phases = np.exp(-1j * _eigvals * t)
        psi_t  = _eigvecs @ (phases * coef)
        return np.abs(psi_t)**2

    return run_ctqw_gcc


# ── GPU methods — exact từ notebook Cell 9 ───────────────────────────────────

def build_gpu_methods(Apro_eigvals, Apro_eigvecs, _pro_src, _pro_dst,
                      N, N_PRO, device=None,
                      t=T_FIXED, n_steps=2, alpha=0.5):
    """
    Factory: trả về dict method_name → fn(psi_batch) → scores_batch.
    Exact từ notebook Cell 9 GPU functions.
    """
    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Exact từ notebook Cell 9
    eigvecs_gpu = torch.tensor(Apro_eigvecs, dtype=torch.complex64).to(device)
    eigvals_gpu = torch.tensor(Apro_eigvals, dtype=torch.float32).to(device)
    eigvecs_H   = eigvecs_gpu.conj().T.contiguous()
    pro_src_t   = torch.tensor(_pro_src, dtype=torch.long).to(device)
    pro_dst_t   = torch.tensor(_pro_dst, dtype=torch.long).to(device)

    def ctqw_step(psi, t_val):
        """Exact từ notebook."""
        phases = torch.exp(-1j * eigvals_gpu * t_val)
        return (eigvecs_gpu @ (phases.unsqueeze(1) * (eigvecs_H @ psi.T))).T

    def psi_to_scores(psi):
        """Exact từ notebook."""
        probs  = psi.abs()**2
        scores = torch.zeros(len(psi), N, dtype=torch.float32, device=device)
        scores[:, pro_dst_t] = probs[:, pro_src_t].float()
        return scores

    def _norm_psi(psi):
        return psi / torch.norm(psi, dim=1, keepdim=True).clamp(min=1e-9)

    # t=0.1 baseline — exact từ notebook Cell 9 methods list
    def ctqw_pro_fn(psi):
        return psi_to_scores(ctqw_step(_norm_psi(psi), t))

    # Driven fixed — exact từ notebook Cell 9 driven_fixed_gpu()
    def driven_fn(psi):
        sn  = torch.norm(psi, dim=1, keepdim=True).clamp(min=1e-9)
        pn  = psi / sn
        ps  = pn.clone()
        for _ in range(n_steps):
            ps = (1 - alpha) * ctqw_step(ps, t) + alpha * pn
            ps = ps / torch.norm(ps, dim=1, keepdim=True).clamp(min=1e-9)
        return psi_to_scores(ps)

    return {
        'ctqw_pro': ctqw_pro_fn,
        'driven':   driven_fn,
        # Expose for custom experiments
        '_ctqw_step':     ctqw_step,
        '_psi_to_scores': psi_to_scores,
        '_norm_psi':      _norm_psi,
        '_device':        device,
        '_eigvecs_gpu':   eigvecs_gpu,
        '_eigvals_gpu':   eigvals_gpu,
        '_eigvecs_H':     eigvecs_H,
        '_pro_src_t':     pro_src_t,
        '_pro_dst_t':     pro_dst_t,
    }


def build_psi_batch(seed_idx_list, N_PRO, device):
    """
    Build psi batch. Exact từ notebook Cell 9 build_psi_batch().
    """
    import torch
    B   = len(seed_idx_list)
    psi = torch.zeros(B, N_PRO, dtype=torch.complex64, device=device)
    for b, sidx in enumerate(seed_idx_list):
        if not sidx: continue
        nr = 1.0 / (len(sidx) ** 0.5)
        for si in sidx: psi[b, si] = nr
    return psi
