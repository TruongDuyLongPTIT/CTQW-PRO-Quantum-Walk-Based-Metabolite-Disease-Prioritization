"""
06_biological_interpretability.py — Biological interpretability analysis.

Dùng NH-CTQW-PRO (full-seed mode) để predict top-20 metabolites
cho 4 selected SMPDB diseases.

Output:
  1. Seeds list cho mỗi disease
  2. Top-20 predictions với HMDB ID
  3. CSV: RESULTS_DIR/biological_interpretability.csv

Classification (ESTABLISHED / POTENTIAL / NOVEL) cần được điền
thủ công sau khi đọc literature — xem cột 'tier' trong CSV,
mặc định là 'UNVERIFIED'.

Usage:
  python 06_biological_interpretability.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pandas as pd

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED,
                    NH_GAMMA, RECON3D_CURRENCY_METABOLITE)
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon,
                   compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups,
                       build_CURRENCY_METABOLITE_set, build_eval_set3)
from methods import make_nh_pro

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# SELECTED DISEASES
# ══════════════════════════════════════════════════════════════════
DISEASES = {
    "LNS":  "Lesch-Nyhan Syndrome (LNS)",
    "AKU":  "Alkaptonuria",
    "MSUD": "Maple Syrup Urine Disease",
    "PKU":  "Phenylketonuria",
}
TOP_K = 20

def hmdb_link(hmdb_id):
    """Generate HMDB metabolite page URL."""
    if not hmdb_id:
        return ""
    return f"https://hmdb.ca/metabolites/{hmdb_id}"

# ══════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════
print('='*65)
print('06 — Biological Interpretability Analysis')
print('='*65)

recon_data   = parse_recon3d()
G_cc, graph_nodes, N, node_idx, A_cc, degrees = build_gcc(recon_data)
met_info     = recon_data['met_info']
pathway_mets = recon_data['pathway_mets']

(G_pro, pro_nodes, N_PRO, idx_pro,
 A_pro, deg_pro, _pro_src, _pro_dst) = build_gpro(G_cc, node_idx, pathway_mets)

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
eval_set3 = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx, CURRENCY_METABOLITE)

# Reverse map: recon_id → HMDB IDs
recon_to_hmdb = {}
for hmdb_id, recon_id in hmdb_to_recon.items():
    recon_to_hmdb.setdefault(recon_id, []).append(hmdb_id)

# Currency metabolite node indices in G_cc (exclude from predictions)
cm_set = set(RECON3D_CURRENCY_METABOLITE)
cm_gcc_idx = set()
for nd, i in node_idx.items():
    nd_b = nd.replace('_c','').replace('_m','').replace('_e','').replace('_x','')
    if nd in cm_set or nd_b in cm_set:
        cm_gcc_idx.add(i)

# ══════════════════════════════════════════════════════════════════
# BUILD NH-CTQW-PRO
# ══════════════════════════════════════════════════════════════════
print(f'Building NH-CTQW-PRO (γ={NH_GAMMA}, t={T_FIXED})...', end=' ', flush=True)
t0 = time.time()
run_nh = make_nh_pro(
    A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
    RECON3D_CURRENCY_METABOLITE, pro_nodes, NH_GAMMA, T_FIXED)
print(f'{time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════
# PREDICT PER DISEASE
# ══════════════════════════════════════════════════════════════════
all_rows = []

for disease_key, disease_name in DISEASES.items():
    if disease_name not in eval_set3:
        print(f'\nWARNING: {disease_name} not found in SMPDB')
        continue

    known_mets  = eval_set3[disease_name]
    valid_seeds = [m for m in known_mets if m in node_idx]
    seed_idx    = {node_idx[s] for s in valid_seeds}

    print(f'\n{"="*65}')
    print(f'DISEASE: {disease_name}  [{disease_key}]')
    print(f'{"="*65}')

    # ── Seeds ─────────────────────────────────────────────────────
    print(f'\nSeeds (n={len(valid_seeds)}) — known disease metabolites:')
    print(f"  {'#':>3}  {'Name':<50}  {'HMDB':>13}  {'HMDB link'}")
    print('  ' + '-'*95)
    for i, s in enumerate(valid_seeds, 1):
        sname = met_info.get(s, {}).get('name', s)
        hmdb  = recon_to_hmdb.get(s, [''])[0]
        print(f"  {i:>3}. {sname:<50}  {hmdb:>13}  {hmdb_link(hmdb)}")

    # ── Predict ───────────────────────────────────────────────────
    scores = run_nh(valid_seeds)

    candidates = []
    for nd, gcc_idx in node_idx.items():
        if gcc_idx in seed_idx:   continue  # mask seeds
        if gcc_idx in cm_gcc_idx: continue  # exclude currency metabolites
        if scores[gcc_idx] < 1e-10: continue
        name  = met_info.get(nd, {}).get('name', nd)
        hmdb  = recon_to_hmdb.get(nd, [''])[0]
        candidates.append({
            'recon_id': nd,
            'name':     name,
            'hmdb_id':  hmdb,
            'score':    float(scores[gcc_idx]),
        })
    candidates.sort(key=lambda x: -x['score'])
    top = candidates[:TOP_K]

    # ── Print predictions ─────────────────────────────────────────
    print(f'\nTop-{TOP_K} predictions (NH-CTQW-PRO, full-seed mode):')
    print(f"  {'Rank':>4}  {'Name':<50}  {'HMDB':>13}  {'Score':>10}")
    print('  ' + '-'*80)

    for i, c in enumerate(top, 1):
        print(f"  {i:>4}. {c['name']:<50}  {c['hmdb_id']:>13}  {c['score']:>10.6f}")

        all_rows.append({
            'disease_key':     disease_key,
            'disease_name':    disease_name,
            'rank':            i,
            'recon_id':        c['recon_id'],
            'name':            c['name'],
            'hmdb_id':         c['hmdb_id'],
            'hmdb_url':        hmdb_link(c['hmdb_id']),
            'score':           c['score'],
            # Tier to be filled manually after literature review:
            # ESTABLISHED = measured in patient samples (cite PMID)
            # POTENTIAL   = mechanistically plausible, indirect evidence
            # NOVEL       = no prior literature connection
            'tier':            'UNVERIFIED',
            'pmid':            '',
            'direct_quote':    '',
            'notes':           '',
        })

# ══════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════
df = pd.DataFrame(all_rows)
out = RESULTS_DIR / 'biological_interpretability.csv'
df.to_csv(out, index=False)

print('\n' + '='*65)
print(f'Saved: {out}  ({len(df)} predictions, {len(DISEASES)} diseases × {TOP_K})')
print('Next: fill tier/pmid/direct_quote/notes columns after literature review.')
print('Done.')
