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


# TABLE 3 — NH-CTQW-PRO

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


def make_netcore_pro(A_pro, deg_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
                     variant='core', core_pro=None,
                     G_pro=None, pro_nodes=None, r=RWR_R):
    if core_pro is None:
        if G_pro is None or pro_nodes is None:
            raise ValueError('Can core_pro, hoac (G_pro, pro_nodes) de tu tinh.')
        core_pro = compute_coreness(G_pro, pro_nodes)

    d = np.asarray(deg_pro, dtype=float)
    k = np.asarray(core_pro, dtype=float)
    d_safe = np.where(d > 0, d, 1.0)

    if   variant == 'core':  w = k
    elif variant == 'diff':  w = 1.0 / ((d - k) + 1.0)
    elif variant == 'ratio': w = k / d_safe
    else:
        raise ValueError(f"variant phai la 'core'|'diff'|'ratio', nhan '{variant}'")

    M = A_pro * w[None, :]                          # trong so theo node DICH
    s = M.sum(axis=1)
    P_nc = M / np.where(s > 0, s, 1.0)[:, None]     # moi hang tong = 1

    _run = make_rwr(P_nc, idx_pro, N_PRO, r=r)      # phan con lai y het PROFANCY
    _src, _dst, _N = _pro_src, _pro_dst, N

    def run_netcore_pro(seed_nodes, _n=_N):
        p = _run(seed_nodes)
        sc = np.zeros(_n)
        sc[_dst] = p[_src]                          # G_pro (2894) -> node_idx (2788)
        return sc

    return run_netcore_pro


def make_dada_ec_pro(P_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst, r=RWR_R):
    _run_rwr = make_rwr(P_pro, idx_pro, N_PRO, r=r)
    p_r0 = make_rwr(P_pro, idx_pro, N_PRO, r=0.0)(list(idx_pro.keys()))
    p_r0_safe = np.where(p_r0 > 1e-15, p_r0, 1e-15)   # chan chia 0, giong
                                                      # make_metaborank_lite_pro
    _src, _dst, _N = _pro_src, _pro_dst, N

    def run_dada_ec_pro(seed_nodes, _n=_N):
        a = _run_rwr(seed_nodes)
        sc = np.zeros(_n)
        sc[_dst] = (a / p_r0_safe)[_src]
        return sc

    return run_dada_ec_pro


#-----Amend IN
# AMEND-IN-LITE — Inflation-Normalization (Boyd et al. 2025, AMEND 2.0)
# Port tu inflate_normalize(), github.com/samboyd0/AMEND, R/RandomWalk.R
# Doi chieu voi R: max|diff| ~ 6e-16

_IN_KF = np.array([1., 10.] + list(np.arange(50., 2001., 50.)))


def _col_norm(X):
    cs = X.sum(0); inv = np.zeros_like(cs); nz = cs != 0; inv[nz] = 1 / cs[nz]
    return X * inv


def _stationary(M):
    n = M.shape[0]; p = np.full(n, 1 / n); delta, step = 1., 0
    while delta > 1e-6 and step <= 100:
        q = M @ p; delta = np.abs(q - p).sum(); p = q; step += 1
    p = np.where(p < 1e-10, 0, p); s = p.sum()
    return np.where(p / s < 1e-10, 0, p / s) if s else p


def _entropy(p):
    return float(np.where(np.round(p, 10) == 0, 0., -p * np.log(np.where(p > 0, p, 1))).sum())


def inflate_normalize(M, verbose=True):
    s0 = _stationary(M); e0 = _entropy(s0); res = np.zeros(len(_IN_KF)); j = -1
    if verbose and np.abs(M @ s0 - s0).sum() > 1e-6:
        print('  [IN] canh bao: phan phoi dung chua hoi tu (do thi gan bipartite?)')
    for i, kf in enumerate(_IN_KF):
        e = _entropy(_stationary(_col_norm(M ** (1 + kf * s0)[:, None])))
        if (e < e0) if i == 0 else (e <= res[i - 1]):
            break
        res[i] = e; j = i
    if j < 0:
        if verbose: print('  [IN] entropy khong tang -> giu nguyen M')
        return M.copy()
    jb = int(np.argmax(res[:j + 1]))
    if verbose: print(f'  [IN] kf* = {_IN_KF[jb]:.0f} | entropy {e0:.4f} -> {res[jb]:.4f}')
    return _col_norm(M ** (1 + _IN_KF[jb] * s0)[:, None])


def make_amend_in_lite(A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst, r=RWR_R, verbose=True):
    P_in = inflate_normalize(_col_norm(np.asarray(A_pro, float)), verbose).T.copy()
    _run = make_rwr(P_in, idx_pro, N_PRO, r=r)
    _src, _dst, _N = _pro_src, _pro_dst, N

    def run_amend_in_lite(seed_nodes, _n=_N):
        sc = np.zeros(_n); sc[_dst] = _run(seed_nodes)[_src]; return sc

    return run_amend_in_lite