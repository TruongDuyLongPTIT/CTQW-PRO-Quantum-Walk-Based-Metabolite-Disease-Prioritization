import json, pickle, re, time, urllib.request, zipfile

import networkx as nx

from config import CACHE_DIR, PATH_HMDB_ZIP
from utils import standardize_hmdb_id

KEGG_BASE       = 'https://rest.kegg.jp'
MIN_INTERVAL_S  = 0.34   # ~3 req/s — KEGG không công bố rate limit cứng, đây là
                         # mức phổ biến cộng đồng dùng để tôn trọng server công cộng.
EXCLUDE_SUBCATS = {'Global and overview maps', 'Chemical structure transformation maps'}

_last_call = [0.0]

def kegg_fetch(path, retries=4):
    url = f'{KEGG_BASE}/{path}'
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode('utf-8')
            _last_call[0] = time.time()
            return text
        except Exception as e:
            _last_call[0] = time.time()
            if attempt == retries - 1:
                print(f'  [WARN] fetch fail {path}: {e}')
                return ''
            time.sleep(1.5 * (attempt + 1))
    return ''


def get_human_metabolism_pathways():
    """BRITE br08901 (Metabolism, trừ overview maps) ∩ list/pathway/hsa."""
    raw = kegg_fetch('get/br:br08901/json')
    tree = json.loads(raw)
    metabolism_root = next(c for c in tree['children'] if c['name'] == 'Metabolism')

    candidates = []
    for subcat in metabolism_root['children']:
        if subcat['name'] in EXCLUDE_SUBCATS:
            continue
        for leaf in subcat.get('children', []):
            m = re.match(r'(\d{5})\s+(.*)', leaf['name'])
            if m:
                candidates.append((m.group(1), m.group(2)))

    raw_hsa = kegg_fetch('list/pathway/hsa')
    hsa_numbers = set()
    for line in raw_hsa.strip().split('\n'):
        if not line:
            continue
        entry = line.split('\t')[0]
        m = re.search(r'hsa(\d{5})', entry)
        if m:
            hsa_numbers.add(m.group(1))

    return [(num, name) for num, name in candidates if num in hsa_numbers]


def get_pathway_reactions(map_number):
    raw = kegg_fetch(f'link/rn/map{map_number}')
    rxn_ids = []
    for line in raw.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            rxn_ids.append(parts[1].replace('rn:', ''))
    return rxn_ids


def fetch_reaction_equations(rxn_ids, batch_size=10):
    """get/rn:R00001+rn:R00002+... → {rxn_id: set(compound_id)} từ EQUATION."""
    rxn_to_cpds = {}
    ids = sorted(set(rxn_ids))
    n_batches = (len(ids) + batch_size - 1) // batch_size
    for bi in range(0, len(ids), batch_size):
        batch = ids[bi:bi + batch_size]
        query = '+'.join(f'rn:{r}' for r in batch)
        raw = kegg_fetch(f'get/{query}')
        for record in raw.split('///'):
            record = record.strip()
            if not record:
                continue
            entry_m = re.search(r'^ENTRY\s+(R\d{5})', record, re.M)
            eq_m    = re.search(r'EQUATION\s+(.*?)(?=\n[A-Z]|\Z)', record, re.S)
            if entry_m and eq_m:
                cpds = set(re.findall(r'C\d{5}', eq_m.group(1)))
                if cpds:
                    rxn_to_cpds[entry_m.group(1)] = cpds
        if (bi // batch_size + 1) % 20 == 0:
            print(f'    equations: {bi // batch_size + 1}/{n_batches} batch...')
    return rxn_to_cpds


def build_kegg_metabolism_data(force=False):
    cache_path = CACHE_DIR / 'kegg_metabolism_raw_v1.pkl'
    if cache_path.exists() and not force:
        with open(cache_path, 'rb') as f:
            raw_data = pickle.load(f)
        print(f'  [cache] {len(raw_data["pathways"])} pathway, '
              f'{len(raw_data["rxn_to_cpds"])} reaction (đã cache)')
    else:
        print('Lấy danh sách pathway chuyển hóa ở người (BRITE br08901 ∩ hsa)...')
        pathways = get_human_metabolism_pathways()
        print(f'  {len(pathways)} pathway (đã loại overview/chemical-structure maps)')

        print('Lấy reaction id theo từng pathway (link/rn/mapNNNNN)...')
        pathway_rxns = {}
        for i, (num, name) in enumerate(pathways):
            pathway_rxns[num] = get_pathway_reactions(num)
            if (i + 1) % 20 == 0:
                print(f'    {i + 1}/{len(pathways)} pathway...')
        all_rxn_ids = sorted({r for rs in pathway_rxns.values() for r in rs})
        print(f'  {len(all_rxn_ids)} reaction id duy nhất (hợp của mọi pathway)')

        print('Lấy EQUATION cho từng reaction (batch 10, get/rn:...+rn:...)...')
        rxn_to_cpds = fetch_reaction_equations(all_rxn_ids)
        print(f'  {len(rxn_to_cpds)}/{len(all_rxn_ids)} reaction có EQUATION parse được')

        raw_data = {'pathways': pathways, 'pathway_rxns': pathway_rxns,
                    'rxn_to_cpds': rxn_to_cpds}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(raw_data, f)

    pathways     = raw_data['pathways']
    pathway_rxns = raw_data['pathway_rxns']
    rxn_to_cpds  = raw_data['rxn_to_cpds']

    edges = set(); all_cpds = set()
    for cpds in rxn_to_cpds.values():
        all_cpds.update(cpds)
        cl = sorted(cpds)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                edges.add((a, b) if a < b else (b, a))

    G = nx.Graph(); G.add_nodes_from(all_cpds); G.add_edges_from(edges)
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    G_cc_kegg = G.subgraph(ccs[0]).copy()

    kegg_nodes    = sorted(G_cc_kegg.nodes())
    N_kegg        = len(kegg_nodes)
    kegg_node_idx = {nd: i for i, nd in enumerate(kegg_nodes)}
    A_cc_kegg     = nx.to_numpy_array(G_cc_kegg, nodelist=kegg_nodes)

    print(f'  G_cc_kegg: {N_kegg} compound (largest CC / {len(all_cpds)} tổng), '
          f'{G_cc_kegg.number_of_edges()} cạnh')

    pathway_mets_kegg = {}
    for num, name in pathways:
        mets = set()
        for rid in pathway_rxns.get(num, []):
            mets.update(rxn_to_cpds.get(rid, set()))
        if mets:
            pathway_mets_kegg[f'{num} {name}'] = mets

    return G_cc_kegg, kegg_node_idx, N_kegg, A_cc_kegg, pathway_mets_kegg


def parse_hmdb_with_kegg(force=False):
    cache_path = CACHE_DIR / 'hmdb_parsed_with_kegg_v1.pkl'
    if cache_path.exists() and not force:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print('Parsing HMDB (~10-20 phút, cần quét lại để lấy thêm field kegg_id)...')
    t0 = time.time()
    NS = '{http://www.hmdb.ca}'
    import xml.etree.ElementTree as ET

    def lname(tag): return tag.split('}', 1)[-1] if '}' in tag else tag
    def gtext(elem, path):
        e = elem.find(NS + path)
        return e.text.strip() if (e is not None and e.text) else ''

    hmdb_metabolites = {}
    with zipfile.ZipFile(PATH_HMDB_ZIP) as zf:
        xml_names = [nm for nm in zf.namelist() if nm.endswith('.xml')]
        with zf.open(xml_names[0]) as fxml:
            for event, elem in ET.iterparse(fxml, events=('end',)):
                if lname(elem.tag) != 'metabolite':
                    continue
                hid_raw = gtext(elem, 'accession')
                if not hid_raw:
                    elem.clear(); continue
                hid    = standardize_hmdb_id(hid_raw)
                status = gtext(elem, 'status')
                cas    = gtext(elem, 'cas_registry_number')
                ik     = gtext(elem, 'inchikey').upper()
                kegg   = gtext(elem, 'kegg_id')
                tax = []
                tx = elem.find(NS + 'taxonomy')
                if tx is not None:
                    for lv in ['kingdom', 'super_class', 'class', 'sub_class', 'direct_parent']:
                        v = gtext(tx, lv)
                        if v: tax.append(v)
                dz_assoc = []; dz_bm = []
                dze = elem.find(NS + 'diseases')
                if dze is not None:
                    for dz in dze.findall(NS + 'disease'):
                        nm = gtext(dz, 'name')
                        if nm: dz_assoc.append({'name': nm, 'omim': gtext(dz, 'omim_id')})
                bme = elem.find(NS + 'biomarkers')
                if bme is not None:
                    for bm in bme.findall(NS + 'biomarker'):
                        nm = gtext(bm, 'name')
                        if nm: dz_bm.append({'name': nm})
                hmdb_metabolites[hid] = {
                    'name': gtext(elem, 'name'), 'inchikey': ik,
                    'status': status, 'cas': cas, 'taxonomy': tax,
                    'diseases_assoc': dz_assoc, 'diseases_biomarker': dz_bm,
                    'kegg_id': kegg,
                }
                elem.clear()
    print(f'  Done in {(time.time() - t0) / 60:.1f} phut, {len(hmdb_metabolites)} metabolite')

    data = {'metabolites': hmdb_metabolites}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def build_hmdb_to_kegg(hmdb_metabolites, kegg_node_idx):
    hmdb_to_kegg = {}
    for hid, m in hmdb_metabolites.items():
        kid = m.get('kegg_id', '')
        if not kid or kid not in kegg_node_idx:
            continue
        hmdb_to_kegg[hid] = kid
        digits = hid[4:].lstrip('0') if hid.startswith('HMDB') else ''
        if digits:
            hmdb_to_kegg.setdefault('HMDB' + digits, kid)
            hmdb_to_kegg.setdefault('HMDB' + digits.zfill(5), kid)
    return hmdb_to_kegg