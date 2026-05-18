"""
graph.py — Build G_cc và G_pro từ Recon3D.
Exact từ notebook Cell 2 và phần augmentation hmdb_to_recon trong Cell 3.
"""
import json, pickle, re, time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from config import PATH_RECON3D, CACHE_DIR
from utils import standardize_hmdb_id


# ── Parse Recon3D ─────────────────────────────────────────────────────────────

def parse_recon3d(force=False):
    """
    Parse Recon3D.json. Exact từ notebook Cell 2.
    Returns: dict với G, met_info, rxn_info, pathway_mets, met_to_rxns, met_to_genes
    """
    cache_path = CACHE_DIR / 'recon3d_parsed_v2.pkl'
    if cache_path.exists() and not force:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    t0 = time.time()
    with open(PATH_RECON3D, 'r', encoding='utf-8') as f:
        recon = json.load(f)
    if isinstance(recon, list):
        for entry in recon:
            if isinstance(entry, dict) and 'metabolites' in entry:
                recon = entry; break

    COMP = {'c','m','e','l','r','g','n','x','i'}
    def strip_comp(mid):
        if '_' in mid:
            base, sfx = mid.rsplit('_', 1)
            if sfx in COMP: return base
        return mid

    met_info = {}
    for met in recon.get('metabolites', []):
        fid = met.get('id', '')
        if not fid: continue
        base = strip_comp(fid)
        if base in met_info: continue
        ann  = met.get('annotation', {}) or {}
        hmdb = ann.get('hmdb', [])
        if isinstance(hmdb, str):   hmdb = [hmdb]
        elif isinstance(hmdb, dict): hmdb = list(hmdb.values())
        hmdb_ids = list(dict.fromkeys(
            standardize_hmdb_id(h) for h in hmdb if h and isinstance(h, str)))
        kegg = ann.get('kegg.compound', [])
        if isinstance(kegg, str):   kegg = [kegg]
        elif isinstance(kegg, dict): kegg = list(kegg.values())
        ik = ann.get('inchikey', '')
        if isinstance(ik, list): ik = ik[0] if ik else ''
        met_info[base] = {
            'name':     met.get('name', base),
            'formula':  met.get('formula', ''),
            'hmdb_ids': hmdb_ids,
            'kegg_ids': [k for k in kegg if k and isinstance(k, str)],
            'inchikey': (ik or '').strip().upper(),
        }

    rxn_info     = {}
    pathway_mets = defaultdict(set)
    met_to_rxns  = defaultdict(set)
    met_to_genes = defaultdict(set)

    for rxn in recon.get('reactions', []):
        rid = rxn.get('id', '')
        if not rid: continue
        mets_raw  = rxn.get('metabolites', {}) or {}
        base_mets = {}
        for fid, coef in mets_raw.items():
            bm = strip_comp(fid)
            base_mets[bm] = base_mets.get(bm, 0) + coef
        genes = set(re.findall(r'\b(\d+)(?:\.\d+)?\b',
                               rxn.get('gene_reaction_rule', '') or ''))
        sub = rxn.get('subsystem', 'Unknown')
        rxn_info[rid] = {'mets': base_mets, 'subsystem': sub, 'genes': genes}
        for mid in base_mets:
            met_to_rxns[mid].add(rid)
            met_to_genes[mid].update(genes)
            if sub and sub != 'Unknown': pathway_mets[sub].add(mid)

    edges = set()
    for rxn in rxn_info.values():
        ms = list(rxn['mets'].keys())
        for i in range(len(ms)):
            for j in range(i+1, len(ms)):
                a, b = ms[i], ms[j]
                if a != b: edges.add((a,b) if a<b else (b,a))

    G = nx.Graph()
    G.add_nodes_from(met_info.keys())
    G.add_edges_from(edges)

    data = {
        'G': G, 'met_info': met_info, 'rxn_info': rxn_info,
        'pathway_mets': dict(pathway_mets),
        'met_to_rxns':  dict(met_to_rxns),
        'met_to_genes': dict(met_to_genes),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f: pickle.dump(data, f)
    print(f'  Parsed in {time.time()-t0:.1f}s → cached')
    return data


# ── Build G_cc ────────────────────────────────────────────────────────────────

def build_gcc(recon_data):
    """Exact từ notebook Cell 2."""
    G   = recon_data['G']
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    G_cc        = G.subgraph(ccs[0]).copy()
    graph_nodes = sorted(G_cc.nodes())   # protected name
    N           = len(graph_nodes)       # protected name
    node_idx    = {nd: i for i, nd in enumerate(graph_nodes)}
    A_cc        = nx.to_numpy_array(G_cc, nodelist=graph_nodes)
    degrees     = A_cc.sum(axis=1)
    return G_cc, graph_nodes, N, node_idx, A_cc, degrees


# ── Build hmdb_to_recon (initial + augmentation) ─────────────────────────────

def build_hmdb_to_recon_initial(met_info, node_idx):
    """
    Initial mapping: HMDB IDs từ Recon3D annotations → Recon3D base IDs.
    Exact từ notebook Cell 2 phần 'HMDB → Recon3D mapping'.
    """
    hmdb_to_recon = {}
    for base_id, info in met_info.items():
        if base_id not in node_idx: continue
        for hid in info['hmdb_ids']:
            hmdb_to_recon[hid] = base_id
            digits = hid[4:].lstrip('0') if hid.startswith('HMDB') else ''
            if digits:
                hmdb_to_recon['HMDB'+digits]          = base_id
                hmdb_to_recon['HMDB'+digits.zfill(5)] = base_id
    return hmdb_to_recon


def augment_hmdb_to_recon(hmdb_to_recon, met_info, node_idx,
                           hmdb_ik_to_id, hmdb_ik_short_to_id,
                           hmdb_name_to_id, hmdb_name_aggr_to_id):
    """
    Augment hmdb_to_recon với IK và name matching.
    Exact từ notebook Cell 3 phần 'Augment hmdb_to_recon'.
    Modifies hmdb_to_recon in-place.

    Returns: (n_aug_ik, n_aug_nm)
    """
    from utils import normalize_name, normalize_chem_aggressive, short_inchikey
    n_aug_ik = 0; n_aug_nm = 0
    mapped = set(hmdb_to_recon.values())

    for base_id, info in met_info.items():
        if base_id not in node_idx or base_id in mapped: continue
        # Try InChIKey first
        ik = info.get('inchikey', '')
        if ik:
            hm = hmdb_ik_to_id.get(ik)
            if not hm:
                sk = short_inchikey(ik)
                hm = hmdb_ik_short_to_id.get(sk) if sk else None
            if hm:
                hmdb_to_recon[hm] = base_id
                mapped.add(base_id)
                n_aug_ik += 1
                continue
        # Try name matching
        nm = normalize_name(info['name'])
        hm = hmdb_name_to_id.get(nm) or hmdb_name_aggr_to_id.get(
             normalize_chem_aggressive(info['name']))
        if hm:
            hmdb_to_recon[hm] = base_id
            mapped.add(base_id)
            n_aug_nm += 1

    return n_aug_ik, n_aug_nm


# ── Build G_pro ───────────────────────────────────────────────────────────────

def build_gpro(G_cc, node_idx, pathway_mets):
    """Exact từ notebook Cell 2."""
    G_pro = G_cc.copy()
    for sub, mets_in_sub in pathway_mets.items():
        valid = [m for m in mets_in_sub if m in node_idx]
        if len(valid) < 2: continue
        pn = f'__PATH__{sub}'
        G_pro.add_node(pn)
        for m in valid: G_pro.add_edge(pn, m)

    pro_nodes = sorted(G_pro.nodes())
    N_PRO     = len(pro_nodes)
    idx_pro   = {nd: i for i, nd in enumerate(pro_nodes)}
    A_pro     = nx.to_numpy_array(G_pro, nodelist=pro_nodes)
    deg_pro   = A_pro.sum(axis=1)

    # _pro_src, _pro_dst: mapping G_pro → G_cc space
    # Used in CTQW scores: scores[_pro_dst] = probs[_pro_src]
    graph_nodes = sorted(G_cc.nodes())
    _pro_src = np.array([idx_pro[nd] for nd in graph_nodes if nd in idx_pro],
                        dtype=np.intp)
    _pro_dst = np.array([node_idx[nd] for nd in graph_nodes if nd in idx_pro],
                        dtype=np.intp)
    return G_pro, pro_nodes, N_PRO, idx_pro, A_pro, deg_pro, _pro_src, _pro_dst


# ── Eigendecomposition ────────────────────────────────────────────────────────

def compute_eigendecomp(A, cache_path=None, force=False):
    """
    numpy.linalg.eigh — symmetric matrix.
    Exact từ notebook Cell 6/7.
    """
    if cache_path and Path(cache_path).exists() and not force:
        d = np.load(cache_path)
        return d['eigvals'], d['eigvecs']
    print(f'  eigh({A.shape[0]}×{A.shape[0]})...')
    t0 = time.time()
    eigvals, eigvecs = np.linalg.eigh(A)
    print(f'  Done in {time.time()-t0:.1f}s')
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, eigvals=eigvals, eigvecs=eigvecs)
    return eigvals, eigvecs


# ── Clean G_pro (cofactors removed) — for ablation study ─────────────────────

def build_clean_gpro(G_pro, node_idx, pathway_mets, cofactors, met_info):
    """
    G_pro với RECON3D_COFACTORS removed.
    Dùng cho ablation study (02_ablation_graph.py).
    cofactors: set of node IDs (RECON3D_COFACTORS).
    """
    G_clean = nx.Graph()
    for nd in G_pro.nodes():
        if nd not in cofactors: G_clean.add_node(nd)
    for u, v in G_pro.edges():
        if u not in cofactors and v not in cofactors:
            G_clean.add_edge(u, v)

    ccs = sorted(nx.connected_components(G_clean), key=len, reverse=True)
    G_cc_cln = G_clean.subgraph(ccs[0]).copy()

    pro_nodes_cln = sorted(G_cc_cln.nodes())
    idx_pro_cln   = {nd: i for i, nd in enumerate(pro_nodes_cln)}
    A_pro_cln     = nx.to_numpy_array(G_cc_cln, nodelist=pro_nodes_cln)
    deg_pro_cln   = A_pro_cln.sum(axis=1)

    gcc_nodes_cln = sorted(nd for nd in pro_nodes_cln if not str(nd).startswith('__PATH__'))
    node_idx_cln  = {nd: i for i, nd in enumerate(gcc_nodes_cln)}
    N_cln         = len(gcc_nodes_cln)

    _pro_src_cln = np.array(
        [idx_pro_cln[nd] for nd in gcc_nodes_cln if nd in idx_pro_cln], dtype=np.intp)
    _pro_dst_cln = np.array(
        [node_idx_cln[nd] for nd in gcc_nodes_cln if nd in idx_pro_cln], dtype=np.intp)

    return (G_cc_cln, pro_nodes_cln, len(pro_nodes_cln), idx_pro_cln,
            A_pro_cln, deg_pro_cln, _pro_src_cln, _pro_dst_cln,
            node_idx_cln, N_cln)
