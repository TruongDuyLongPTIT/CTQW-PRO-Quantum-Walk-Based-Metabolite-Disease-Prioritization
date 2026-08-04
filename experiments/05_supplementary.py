"""
  Supplementary S1 — γ grid search (NH-CTQW-PRO)
  Supplementary S2 — CTQW-PRO t grid search
"""
import pandas as pd
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import warnings; warnings.filterwarnings('ignore')
from graph import (parse_recon3d, build_gcc, build_gpro, build_hmdb_to_recon_initial, augment_hmdb_to_recon, compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups, build_CURRENCY_METABOLITE_set, build_eval_set1, build_eval_set3)

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED, NH_GAMMA, RECON3D_CURRENCY_METABOLITE)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------SETUP —-----------
def setup_pipeline():
    print('Setup pipeline...')
    recon_data   = parse_recon3d()
    G_cc, _, N, node_idx, A_cc, _ = build_gcc(recon_data)
    met_info     = recon_data['met_info']
    pathway_mets = recon_data['pathway_mets']

    (G_pro, pro_nodes, N_PRO, idx_pro, A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

    eigvals, eigvecs = compute_eigendecomp(A_pro, CACHE_DIR / 'gpro_eigdecomp.npz')

    hmdb_data        = parse_hmdb()
    hmdb_metabolites = hmdb_data['metabolites']
    hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)
    hmdb_to_recon    = build_hmdb_to_recon_initial(met_info, node_idx)
    augment_hmdb_to_recon(
        hmdb_to_recon, met_info, node_idx,
        hmdb_lookups['ik_to_id'], hmdb_lookups['ikshort_to_id'],
        hmdb_lookups['name_to_id'], hmdb_lookups['name_aggr_to_id'])
    CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)
    eval_set1, _ = build_eval_set1(hmdb_metabolites, hmdb_lookups, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)
    eval_set3    = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)
    print(f'  HMDB+CTD: {len(eval_set1)}, SMPDB: {len(eval_set3)}')

    return dict(
        N=N, node_idx=node_idx, N_PRO=N_PRO, idx_pro=idx_pro,
        A_pro=A_pro, deg_pro=deg_pro, pro_nodes=pro_nodes,
        eigvals=eigvals, eigvecs=eigvecs,
        _pro_src=_pro_src, _pro_dst=_pro_dst,
        eval_set1=eval_set1, eval_set3=eval_set3,
        CURRENCY_METABOLITE=CURRENCY_METABOLITE,
    )

# ------------------S1 — γ GRID SEARCH (NH-CTQW-PRO)-------------------
GAMMA_GRID_S1 = [10, 15, 22, 30, 40, 50, 60, 80, 150, 349]

def run_s1_experiment(ctx):
    """Chạy γ grid search nếu CSV chưa có."""
    from methods import make_nh_pro, make_ctqw_pro
    from evaluation import run_loo_eval

    N, node_idx   = ctx['N'], ctx['node_idx']
    N_PRO         = ctx['N_PRO']
    idx_pro       = ctx['idx_pro']
    A_pro         = ctx['A_pro']
    pro_nodes     = ctx['pro_nodes']
    eigvals       = ctx['eigvals']
    eigvecs       = ctx['eigvecs']
    _pro_src      = ctx['_pro_src']
    _pro_dst      = ctx['_pro_dst']
    eval_set1     = ctx['eval_set1']
    eval_set3     = ctx['eval_set3']

    # CTQW baseline
    run_ctqw = make_ctqw_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)
    ctqw_fn = lambda seeds: run_ctqw(seeds, [T_FIXED])[T_FIXED]

    base_rows = []
    for label, dset in [('HMDB+CTD', eval_set1), ('SMPDB', eval_set3)]:
        df = run_loo_eval(dset, ctqw_fn, node_idx, N, label=f'CTQW/{label}')
        if df is not None:
            tmp = df.copy(); tmp['gamma'] = 'baseline'; tmp['dataset'] = label
            base_rows.append(tmp)
    if base_rows:
        pd.concat(base_rows).to_csv(
            RESULTS_DIR / 'grid_gamma_baseline.csv', index=False)

    # γ grid
    all_rows = []
    for gamma in GAMMA_GRID_S1:
        t0 = time.time()
        print(f'  γ={gamma}...', end=' ', flush=True)
        run_nh = make_nh_pro(
            A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
            RECON3D_CURRENCY_METABOLITE, pro_nodes, float(gamma), T_FIXED)
        print(f'{time.time()-t0:.1f}s')
        for label, dset in [('HMDB+CTD', eval_set1), ('SMPDB', eval_set3)]:
            df = run_loo_eval(dset, run_nh, node_idx, N,
                              label=f'NH(γ={gamma})/{label}')
            if df is not None and not df.empty:
                tmp = df.copy(); tmp['gamma'] = gamma; tmp['dataset'] = label
                all_rows.append(tmp)

    if all_rows:
        pd.concat(all_rows).to_csv(
            RESULTS_DIR / 'grid_gamma_search.csv', index=False)


def print_s1():
    """Supplementary S1 — γ grid search, in bảng MRR theo γ (không vẽ hình)."""
    print('\n── Supplementary S1: γ grid search ──')

    csv_path = RESULTS_DIR / 'grid_gamma_search.csv'
    base_path = RESULTS_DIR / 'grid_gamma_baseline.csv'

    if not csv_path.exists():
        print('  CSV not found — running experiment first...')
        ctx = setup_pipeline()
        run_s1_experiment(ctx)

    df = pd.read_csv(csv_path)
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce')
    df = df.dropna(subset=['gamma'])
    df_base = pd.read_csv(base_path) if base_path.exists() else None

    gammas = sorted(df['gamma'].unique())
    for label in ['HMDB+CTD', 'SMPDB']:
        sub = df[df['dataset'] == label]
        base_mrr = None
        if df_base is not None:
            b = df_base[df_base['dataset'] == label]
            if not b.empty: base_mrr = float(b['mrr'].mean())

        print(f'\n  [{label}]  {"γ":>6}   {"MRR":>8}')
        print('  ' + '-'*22)
        if base_mrr is not None:
            print(f'  {"CTQW-PRO":>6}   {base_mrr:>8.4f}   (baseline, γ→∞)')
        for g in gammas:
            mrr = sub[sub['gamma'] == g]['mrr'].mean()
            mark = '  ← paper (γ=k̄)' if g == NH_GAMMA else ''
            print(f'  {g:>6.0f}   {mrr:>8.4f}{mark}')


# ════════════════════════════════════════════════════════════════
# S2 — t GRID SEARCH (CTQW-PRO, HMDB+CTD)
# ════════════════════════════════════════════════════════════════
T_GRID_S2 = [0.05, 0.1, 0.2, 0.5, 0.8]  # grid dùng để chọn T_FIXED trong paper (Mục 4.x)

def run_s2_experiment(ctx):
    # Chạy t grid search (CTQW-PRO, HMDB+CTD) nếu CSV chưa có.
    from methods import make_ctqw_pro
    from evaluation import run_loo_eval

    N, node_idx = ctx['N'], ctx['node_idx']
    N_PRO       = ctx['N_PRO']
    idx_pro     = ctx['idx_pro']
    eigvals     = ctx['eigvals']
    eigvecs     = ctx['eigvecs']
    _pro_src    = ctx['_pro_src']
    _pro_dst    = ctx['_pro_dst']
    eval_set1   = ctx['eval_set1']

    run_ctqw = make_ctqw_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)

    all_rows = []
    for t in T_GRID_S2:
        ctqw_fn = lambda seeds, _t=t: run_ctqw(seeds, [_t])[_t]
        df = run_loo_eval(eval_set1, ctqw_fn, node_idx, N, label=f'CTQW-PRO(t={t})/HMDB+CTD')
        if df is not None and not df.empty:
            tmp = df.copy(); tmp['t'] = t
            all_rows.append(tmp)

    if all_rows:
        pd.concat(all_rows).to_csv(RESULTS_DIR / 'grid_t_search.csv', index=False)


def print_s2():
    # Supplementary S2 — t grid search
    print('\n── Supplementary S2: t grid search ──')

    csv_path = RESULTS_DIR / 'grid_t_search.csv'
    if not csv_path.exists():
        print('  CSV not found — running experiment first...')
        ctx = setup_pipeline()
        run_s2_experiment(ctx)

    df_t = pd.read_csv(csv_path)
    t_vals = sorted(df_t['t'].unique())
    mrrs   = {t: df_t[df_t['t'] == t]['mrr'].mean() for t in t_vals}
    best_t = max(mrrs, key=mrrs.get)

    print(f'\n  [CTQW-PRO, HMDB+CTD]  {"t":>6}   {"MRR":>8}')
    print('  ' + '-'*24)
    for t in t_vals:
        mark = '  ← selected' if t == best_t else ''
        print(f'  {t:>6}   {mrrs[t]:>8.4f}{mark}')


# -----MAIN--------
print_s1()
print_s2()
