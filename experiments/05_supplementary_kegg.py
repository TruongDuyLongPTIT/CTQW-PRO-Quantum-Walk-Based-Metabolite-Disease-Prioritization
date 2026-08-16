import pandas as pd
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import warnings; warnings.filterwarnings('ignore')
from kegg_graph import build_kegg_metabolism_data, parse_hmdb_with_kegg, build_hmdb_to_kegg
from graph import build_gpro, compute_eigendecomp
from eval_sets import build_hmdb_lookups, build_CURRENCY_METABOLITE_set, build_eval_set1, build_eval_set3

from config import RESULTS_DIR, CACHE_DIR, T_FIXED, NH_GAMMA_KEGG, KEGG_CURRENCY_METABOLITE

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def setup_pipeline():
    print('Setup pipeline (KEGG)...')
    G_cc, node_idx, N, _, pathway_mets = build_kegg_metabolism_data()

    (G_pro, pro_nodes, N_PRO, idx_pro, A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

    eigvals, eigvecs = compute_eigendecomp(A_pro, CACHE_DIR / 'gpro_kegg_eigdecomp.npz')

    hmdb_data        = parse_hmdb_with_kegg()
    hmdb_metabolites = hmdb_data['metabolites']
    hmdb_lookups     = build_hmdb_lookups(hmdb_metabolites)
    hmdb_to_kegg     = build_hmdb_to_kegg(hmdb_metabolites, node_idx)
    CURRENCY_METABOLITE = build_CURRENCY_METABOLITE_set(hmdb_metabolites)
    eval_set1, _ = build_eval_set1(hmdb_metabolites, hmdb_lookups, hmdb_to_kegg, node_idx, CURRENCY_METABOLITE,
                                   currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)
    eval_set3    = build_eval_set3(hmdb_metabolites, hmdb_to_kegg, node_idx, CURRENCY_METABOLITE,
                                   currency_metabolite_ids=KEGG_CURRENCY_METABOLITE)
    print(f'  HMDB+CTD: {len(eval_set1)}, SMPDB: {len(eval_set3)}')

    return dict(
        N=N, node_idx=node_idx, N_PRO=N_PRO, idx_pro=idx_pro,
        A_pro=A_pro, deg_pro=deg_pro, pro_nodes=pro_nodes,
        eigvals=eigvals, eigvecs=eigvecs,
        _pro_src=_pro_src, _pro_dst=_pro_dst,
        eval_set1=eval_set1, eval_set3=eval_set3,
        CURRENCY_METABOLITE=CURRENCY_METABOLITE,
    )

# ------------------S1 — γ GRID SEARCH (NH-CTQW-PRO, KEGG)-------------------
GAMMA_GRID_S1 = [7, 11, 16, 22, 29, 36, 44, 58, 109, 254]

def run_s1_experiment(ctx):
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

    run_ctqw = make_ctqw_pro(eigvals, eigvecs, idx_pro, N, N_PRO, _pro_src, _pro_dst)
    ctqw_fn = lambda seeds: run_ctqw(seeds, T_FIXED)

    base_rows = []
    for label, dset in [('HMDB+CTD', eval_set1), ('SMPDB', eval_set3)]:
        df = run_loo_eval(dset, ctqw_fn, node_idx, N, label=f'CTQW/{label}')
        if df is not None:
            tmp = df.copy(); tmp['gamma'] = 'baseline'; tmp['dataset'] = label
            base_rows.append(tmp)
    if base_rows:
        pd.concat(base_rows).to_csv(
            RESULTS_DIR / 'grid_gamma_kegg_baseline.csv', index=False)

    all_rows = []
    for gamma in GAMMA_GRID_S1:
        t0 = time.time()
        print(f'  γ={gamma}...', end=' ', flush=True)
        run_nh = make_nh_pro(
            A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
            KEGG_CURRENCY_METABOLITE, pro_nodes, float(gamma), T_FIXED)
        print(f'{time.time()-t0:.1f}s')
        for label, dset in [('HMDB+CTD', eval_set1), ('SMPDB', eval_set3)]:
            df = run_loo_eval(dset, run_nh, node_idx, N,
                              label=f'NH(γ={gamma})/{label}')
            if df is not None and not df.empty:
                tmp = df.copy(); tmp['gamma'] = gamma; tmp['dataset'] = label
                all_rows.append(tmp)

    if all_rows:
        pd.concat(all_rows).to_csv(
            RESULTS_DIR / 'grid_gamma_kegg_search.csv', index=False)


def print_s1():
    print('\n── Supplementary S1 (KEGG): γ grid search ──')

    csv_path = RESULTS_DIR / 'grid_gamma_kegg_search.csv'
    base_path = RESULTS_DIR / 'grid_gamma_kegg_baseline.csv'

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
            mark = '  ← paper (γ=k̄)' if g == NH_GAMMA_KEGG else ''
            print(f'  {g:>6.0f}   {mrr:>8.4f}{mark}')


# ════════════════════════════════════════════════════════════════
# S2 — t GRID SEARCH (CTQW-PRO, HMDB+CTD, KEGG)
# ════════════════════════════════════════════════════════════════
T_GRID_S2 = [0.05, 0.1, 0.2, 0.5, 0.8]

def run_s2_experiment(ctx):
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
        ctqw_fn = lambda seeds, _t=t: run_ctqw(seeds, _t)
        df = run_loo_eval(eval_set1, ctqw_fn, node_idx, N, label=f'CTQW-PRO(t={t})/HMDB+CTD')
        if df is not None and not df.empty:
            tmp = df.copy(); tmp['t'] = t
            all_rows.append(tmp)

    if all_rows:
        pd.concat(all_rows).to_csv(RESULTS_DIR / 'grid_t_kegg_search.csv', index=False)


def print_s2():
    print('\n── Supplementary S2 (KEGG): t grid search ──')

    csv_path = RESULTS_DIR / 'grid_t_kegg_search.csv'
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


print_s1()
print_s2()
