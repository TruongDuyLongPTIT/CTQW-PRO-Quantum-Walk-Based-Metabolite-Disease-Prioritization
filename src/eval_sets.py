"""
eval_sets.py — Build 3 evaluation sets.
Exact từ notebook Cells 3 (HMDB+CTD), 4 (MarkerDB), 5 (SMPDB).

CRITICAL correctness notes:
- eval_set1/2/3: filter `base in RECON3D_CURRENCY_METABOLITE or normalize_name(name) in COFACTORS`
  (ĐỒNG NHẤT cho cả 3 eval set — patch 2026-07: trước đây eval_set3 chỉ lọc theo
  RECON3D_CURRENCY_METABOLITE bằng node ID, không lọc theo tên COFACTORS như set1/2. Đã kiểm
  chứng bằng thực nghiệm: áp thêm name-filter cho SMPDB không thay đổi kết quả
  (Jaccard=1.0 trên toàn bộ 153/153 bệnh, 0 metabolite bị loại thêm) — nên patch này
  không ảnh hưởng số liệu đã báo cáo trước đó, chỉ để đồng bộ logic cho rõ ràng.
- hmdb_to_recon: shared mutable dict, augmented in Cell 3
- CTD: split(',') + pts[5] for DirectEvidence = 'marker'
- SMPDB: extract to /tmp/smpdb/ for zip files
"""
import gzip, pickle, re, shutil, time, zipfile
from collections import defaultdict
from itertools import combinations as _combinations
from pathlib import Path

import pandas as pd

from config import (
    PATH_HMDB_ZIP, PATH_CTD, PATH_MARKERDB,
    PATH_SMPDB_PW, PATH_SMPDB_MET,
    SMPDB_MET_DIR, SMPDB_PW_DIR,
    CACHE_DIR, BASE_DIR,
    RECON3D_CURRENCY_METABOLITE, COFACTORS_FALLBACK,
    GENERIC_DISEASES, ALLOWED_STATUSES, MIN_METS,
)
from utils import standardize_hmdb_id, normalize_name, normalize_chem_aggressive


# ── Parse HMDB + build lookups — exact từ notebook Cell 3 ────────────────────

def parse_hmdb(force=False):
    """Parse hmdb_metabolites.zip. Exact từ notebook Cell 3."""
    cache_path = CACHE_DIR / 'hmdb_parsed_v3.pkl'
    if cache_path.exists() and not force:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    t0 = time.time()
    print('  Parsing HMDB (~10-20 min)...')
    NS = '{http://www.hmdb.ca}'

    import xml.etree.ElementTree as ET

    def lname(tag): return tag.split('}',1)[-1] if '}' in tag else tag
    def gtext(elem, path):
        e = elem.find(NS+path)
        return e.text.strip() if (e is not None and e.text) else ''

    hmdb_metabolites = {}
    with zipfile.ZipFile(PATH_HMDB_ZIP) as zf:
        xml_names = [nm for nm in zf.namelist() if nm.endswith('.xml')]
        with zf.open(xml_names[0]) as fxml:
            for event, elem in ET.iterparse(fxml, events=('end',)):
                if lname(elem.tag) != 'metabolite': continue
                hid_raw = gtext(elem, 'accession')
                if not hid_raw: elem.clear(); continue
                hid    = standardize_hmdb_id(hid_raw)
                status = gtext(elem, 'status')
                cas    = gtext(elem, 'cas_registry_number')
                ik     = gtext(elem, 'inchikey').upper()
                tax    = []
                tx = elem.find(NS+'taxonomy')
                if tx is not None:
                    for lv in ['kingdom','super_class','class','sub_class','direct_parent']:
                        v = gtext(tx, lv)
                        if v: tax.append(v)
                dz_assoc = []; dz_bm = []
                dze = elem.find(NS+'diseases')
                if dze is not None:
                    for dz in dze.findall(NS+'disease'):
                        nm = gtext(dz,'name')
                        if nm: dz_assoc.append({'name':nm,'omim':gtext(dz,'omim_id')})
                bme = elem.find(NS+'biomarkers')
                if bme is not None:
                    for bm in bme.findall(NS+'biomarker'):
                        nm = gtext(bm,'name')
                        if nm: dz_bm.append({'name':nm})
                hmdb_metabolites[hid] = {
                    'name': gtext(elem,'name'), 'inchikey': ik,
                    'status': status, 'cas': cas, 'taxonomy': tax,
                    'diseases_assoc': dz_assoc, 'diseases_biomarker': dz_bm,
                }
                elem.clear()

    data = {'metabolites': hmdb_metabolites}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f: pickle.dump(data, f)
    print(f'  Done in {time.time()-t0:.1f}s → cached')
    return data


def build_hmdb_lookups(hmdb_metabolites):
    """
    Build lookup dicts.
    Exact từ notebook Cell 3 'Build lookup indices'.
    """
    from utils import short_inchikey
    hmdb_name_to_id      = {}
    hmdb_name_aggr_to_id = {}
    hmdb_ik_to_id        = {}
    hmdb_ik_short_to_id  = {}
    hmdb_cas_to_id       = {}

    for hid, m in hmdb_metabolites.items():
        nm = normalize_name(m['name'])
        if nm: hmdb_name_to_id.setdefault(nm, hid)
        nma = normalize_chem_aggressive(m['name'])
        if nma: hmdb_name_aggr_to_id.setdefault(nma, hid)
        ik = m['inchikey']
        if ik:
            hmdb_ik_to_id.setdefault(ik, hid)
            sk = short_inchikey(ik)
            if sk: hmdb_ik_short_to_id.setdefault(sk, hid)
        cas = m.get('cas', '')
        if cas: hmdb_cas_to_id[cas] = hid

    return {
        'name_to_id':      hmdb_name_to_id,
        'name_aggr_to_id': hmdb_name_aggr_to_id,
        'ik_to_id':        hmdb_ik_to_id,
        'ikshort_to_id':   hmdb_ik_short_to_id,
        'cas_to_id':       hmdb_cas_to_id,
    }


def build_cofactors_set(hmdb_metabolites):
    """
    Build COFACTORS name-based set.
    Exact từ notebook Cell 3 'Cofactor set'.
    """
    COFACTORS = set(COFACTORS_FALLBACK)
    for hid, m in hmdb_metabolites.items():
        tx = ' '.join(m['taxonomy']).lower()
        if any(kw in tx for kw in ['cofactor', 'coenzyme']):
            nm = normalize_name(m['name'])
            if nm: COFACTORS.add(nm)
    return COFACTORS


# ── Eval set 1: HMDB + CTD — exact từ notebook Cell 3 ────────────────────────

def build_eval_set1(hmdb_metabolites, hmdb_lookups, hmdb_to_recon,
                    node_idx, COFACTORS, min_mets=MIN_METS):
    """
    Exact từ notebook Cell 3.

    CRITICAL:
    - HMDB filter: `base in RECON3D_CURRENCY_METABOLITE or normalize_name(m['name']) in COFACTORS`
    - CTD filter:  `base in RECON3D_CURRENCY_METABOLITE` + `normalize_name(chem) in COFACTORS`
    - CTD parsing: split(',') + pts[5] DirectEvidence contains 'marker'
    """
    hmdb_name_to_id      = hmdb_lookups['name_to_id']
    hmdb_name_aggr_to_id = hmdb_lookups['name_aggr_to_id']
    hmdb_cas_to_id       = hmdb_lookups['cas_to_id']
    disease_name_canonical = {}

    # ── HMDB associations ──
    gt_hmdb = defaultdict(set)
    for hid, m in hmdb_metabolites.items():
        if (m['status'] or '').strip().lower() not in ALLOWED_STATUSES: continue
        base = hmdb_to_recon.get(hid)
        if not base or base not in node_idx: continue
        # Exact cofactor check from notebook
        if base in RECON3D_CURRENCY_METABOLITE or normalize_name(m['name']) in COFACTORS:
            continue
        for dz in m['diseases_assoc'] + m['diseases_biomarker']:
            dn = normalize_name(dz['name'])
            if dn:
                gt_hmdb[dn].add(base)
                disease_name_canonical[dn] = dz['name']

    # ── Parse CTD — exact từ notebook Cell 3 ──
    cache_ctd = CACHE_DIR / 'ctd_parsed_cas.pkl'
    if cache_ctd.exists():
        with open(cache_ctd, 'rb') as f:
            ctd_data = pickle.load(f)
    else:
        t0 = time.time(); rows = []; hdr = None
        with gzip.open(PATH_CTD, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line: continue
                if line.startswith('#'):
                    if 'ChemicalName' in line and 'DiseaseName' in line:
                        hdr = line.lstrip('# ').split(',')
                    continue
                if hdr is None: hdr = line.split(','); continue
                pts = line.split(',')
                if len(pts) < 6: continue
                # Exact: pts[5] = DirectEvidence column
                if 'marker' not in pts[5].strip().lower(): continue
                rows.append({'ChemicalName': pts[0].strip(),
                             'CAS':          pts[2].strip(),
                             'DiseaseName':  pts[3].strip()})
        ctd_data = {'df': pd.DataFrame(rows)}
        with open(cache_ctd, 'wb') as f: pickle.dump(ctd_data, f)
        print(f'  CTD parsed in {time.time()-t0:.1f}s')

    df_ctd    = ctd_data['df']
    ctd_cache = {}

    def ctd_map(chem, cas=''):
        key = (chem, cas)
        if key in ctd_cache: return ctd_cache[key]
        if cas and cas in hmdb_cas_to_id:
            base = hmdb_to_recon.get(hmdb_cas_to_id[cas])
            if base and base in node_idx:
                ctd_cache[key] = base; return base
        nm = normalize_name(chem)
        hm = hmdb_name_to_id.get(nm) or hmdb_name_aggr_to_id.get(
             normalize_chem_aggressive(chem))
        if hm:
            base = hmdb_to_recon.get(hm)
            if base and base in node_idx:
                ctd_cache[key] = base; return base
        ctd_cache[key] = None; return None

    gt_ctd = defaultdict(set)
    for _, row in df_ctd.iterrows():
        base = ctd_map(row['ChemicalName'], row.get('CAS', ''))
        if not base or base in RECON3D_CURRENCY_METABOLITE: continue
        # Exact from notebook
        if normalize_name(row['ChemicalName']) in COFACTORS: continue
        dn = normalize_name(row['DiseaseName'])
        if not dn: continue
        gt_ctd[dn].add(base)
        if dn not in disease_name_canonical:
            disease_name_canonical[dn] = row['DiseaseName']

    # Merge
    gt_all = defaultdict(set)
    for d, m in gt_hmdb.items(): gt_all[d].update(m)
    for d, m in gt_ctd.items():  gt_all[d].update(m)

    eval_set1 = {
        disease_name_canonical[d]: sorted(m)
        for d, m in gt_all.items()
        if len(m) >= min_mets and d not in GENERIC_DISEASES
    }
    return eval_set1, disease_name_canonical


# ── Eval set 2: MarkerDB — exact từ notebook Cell 4 ──────────────────────────

def build_eval_set2(hmdb_metabolites, hmdb_lookups, hmdb_to_recon,
                    node_idx, COFACTORS, disease_name_canonical,
                    min_mets=MIN_METS):
    """Exact từ notebook Cell 4."""
    from utils import short_inchikey
    hmdb_name_to_id      = hmdb_lookups['name_to_id']
    hmdb_name_aggr_to_id = hmdb_lookups['name_aggr_to_id']
    hmdb_ik_to_id        = hmdb_lookups['ik_to_id']
    hmdb_ik_short_to_id  = hmdb_lookups['ikshort_to_id']

    cache_marker = CACHE_DIR / 'markerdb_parsed_v4.pkl'
    if cache_marker.exists():
        with open(cache_marker, 'rb') as f:
            markerdb_data = pickle.load(f)
        if not markerdb_data.get('markers'):
            markerdb_data = None
    else:
        markerdb_data = None

    if markerdb_data is None:
        import xml.etree.ElementTree as ET
        t0 = time.time()
        with open(PATH_MARKERDB, 'r', encoding='utf-8') as f: raw = f.read()
        raw = re.sub(r'<\?xml[^?]*\?>', '', raw).strip()
        root = None
        try:
            root = ET.fromstring(f'<root>{raw}</root>')
        except ET.ParseError:
            chunks = []
            for tag in ('chemical','marker','biomarker'):
                chunks.extend(re.findall(rf'<{tag}[^>]*>.*?</{tag}>', raw, re.DOTALL))
            try: root = ET.fromstring(f"<root>{''.join(set(chunks))}</root>")
            except: pass

        def gmk(elem, tag):
            e = elem.find(tag)
            return e.text.strip() if (e is not None and e.text) else ''

        markers = []
        if root is not None:
            celems = [e for e in root.iter()
                      if e.tag.split('}',1)[-1].lower() in ('chemical','marker','biomarker')]
            for chem in celems:
                name = gmk(chem,'name') or gmk(chem,'chemical_name')
                if not name:
                    for sub in chem:
                        if sub.tag.split('}',1)[-1].lower()=='name' and sub.text:
                            name=sub.text.strip(); break
                hmdb_id = ''
                for sub in chem.iter():
                    t2 = sub.tag.split('}',1)[-1].lower()
                    if t2 in ('hmdb_id','hmdb','hmdb_accession') and sub.text:
                        hmdb_id=standardize_hmdb_id(sub.text.strip()); break
                ik2 = ''
                for sub in chem.iter():
                    t2 = sub.tag.split('}',1)[-1].lower()
                    if t2 in ('inchikey','inchi_key') and sub.text:
                        ik2=sub.text.strip().upper(); break
                conds = []
                for cont in chem.iter():
                    t2 = cont.tag.split('}',1)[-1].lower()
                    if t2 in ('conditions','condition_associations','associated_conditions',
                              'condition_association','associated_diseases','diseases'):
                        for c in cont:
                            ct = c.tag.split('}',1)[-1].lower()
                            if ct in ('condition','disease','association'):
                                cn = (gmk(c,'name') or gmk(c,'condition_name') or
                                      gmk(c,'disease_name') or
                                      (c.text.strip() if c.text else ''))
                                if cn: conds.append(cn)
                    if t2 == 'condition':
                        cn = gmk(cont,'name') or (cont.text.strip() if cont.text else '')
                        if cn: conds.append(cn)
                if name:
                    markers.append({'name':name,'hmdb_id':hmdb_id,
                                    'inchikey':ik2,
                                    'conditions':list(dict.fromkeys(conds))})
        markerdb_data = {'markers': markers}
        with open(cache_marker,'wb') as f: pickle.dump(markerdb_data, f)
        print(f'  Parsed {len(markers)} markers in {time.time()-t0:.1f}s')

    markers = markerdb_data['markers']
    mk_raw  = defaultdict(set)
    for m in markers:
        if not m['conditions']: continue
        base = None
        if m['hmdb_id']: base = hmdb_to_recon.get(m['hmdb_id'])
        if not base and m['inchikey']:
            hm = hmdb_ik_to_id.get(m['inchikey'])
            if not hm:
                sk = short_inchikey(m['inchikey'])
                hm = hmdb_ik_short_to_id.get(sk) if sk else None
            if hm: base = hmdb_to_recon.get(hm)
        if not base:
            nm = normalize_name(m['name'])
            hm = hmdb_name_to_id.get(nm) or hmdb_name_aggr_to_id.get(
                 normalize_chem_aggressive(m['name']))
            if hm: base = hmdb_to_recon.get(hm)
        if not base or base not in node_idx: continue
        if base in RECON3D_CURRENCY_METABOLITE or normalize_name(m['name']) in COFACTORS:
            continue
        for cond in m['conditions']:
            cn = normalize_name(cond)
            if cn and cn not in GENERIC_DISEASES:
                mk_raw[cn].add(base)
                if cn not in disease_name_canonical:
                    disease_name_canonical[cn] = cond

    eval_set2 = {
        disease_name_canonical[d]: sorted(ms)
        for d, ms in mk_raw.items()
        if len(ms) >= min_mets
    }
    return eval_set2


# ── SMPDB helpers — exact từ notebook Cell 5 ─────────────────────────────────

TMP_SMPDB = Path('/tmp/smpdb')

def _get_smpdb_pw():
    """Exact từ notebook Cell 5 _get_smpdb_pw()."""
    for cand in [BASE_DIR/'smpdb_pathways.csv', SMPDB_PW_DIR]:
        if cand.is_file(): return cand
    zip_path = PATH_SMPDB_PW
    if zip_path.exists():
        TMP_SMPDB.mkdir(exist_ok=True)
        out = TMP_SMPDB/'smpdb_pathways.csv'
        if not out.exists():
            with zipfile.ZipFile(zip_path) as zf:
                csvs = [nm for nm in zf.namelist() if nm.endswith('.csv')]
                zf.extract(csvs[0], TMP_SMPDB)
                extracted = TMP_SMPDB/csvs[0]
                if extracted != out: extracted.rename(out)
        return out
    raise FileNotFoundError('smpdb_pathways.csv not found')


def _get_smpdb_met_dir():
    """Exact từ notebook Cell 5 _get_smpdb_met_dir()."""
    if SMPDB_MET_DIR.is_dir() and any(SMPDB_MET_DIR.glob('SMP*.csv')):
        return SMPDB_MET_DIR
    zip_path = PATH_SMPDB_MET
    if zip_path.exists():
        TMP_SMPDB.mkdir(exist_ok=True)
        met_dir = TMP_SMPDB/'mets'
        if not met_dir.exists() or not any(met_dir.glob('SMP*.csv')):
            met_dir.mkdir(exist_ok=True)
            print('  Extracting SMPDB metabolites...')
            with zipfile.ZipFile(zip_path) as zf:
                csv_files = [nm for nm in zf.namelist() if nm.endswith('.csv')]
                if len(csv_files) == 1:
                    # Single large CSV — split by pathway
                    zf.extract(csv_files[0], TMP_SMPDB)
                    single = TMP_SMPDB/csv_files[0]
                    df_s   = pd.read_csv(single, low_memory=False)
                    mc     = {c.lower().strip():c for c in df_s.columns}
                    sc     = next(
                        (mc[k] for k in mc
                         if 'pathway_subject_db_id' in k
                         or ('smpdb' in k and 'id' in k)), None)
                    if sc:
                        for sid, grp in df_s.groupby(sc):
                            grp.to_csv(met_dir/f'{sid}_metabolites.csv', index=False)
                    else:
                        shutil.copy(single, met_dir/'all_metabolites.csv')
                else:
                    from tqdm.auto import tqdm
                    for nm in tqdm(csv_files, desc='Extract SMPDB', leave=False):
                        zf.extract(nm, met_dir)
        return met_dir
    raise FileNotFoundError('smpdb_metabolites not found')


# ── Eval set 3: SMPDB — exact từ notebook Cell 5 ─────────────────────────────

def build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx,
                    COFACTORS, min_mets=MIN_METS):
    """
    Exact từ notebook Cell 5.

    Cofactor filter — ĐỒNG NHẤT với eval_set1/2 (patch 2026-07):
        `base in RECON3D_CURRENCY_METABOLITE or normalize_name(name) in COFACTORS`
    Trước patch này, eval_set3 chỉ lọc theo RECON3D_CURRENCY_METABOLITE (node ID),
    không lọc theo tên (COFACTORS) như eval_set1/2. Đã kiểm chứng thực
    nghiệm (04_check_smpdb_cofactor_filter.py): áp thêm name-filter cho
    SMPDB cho Jaccard=1.0 trên 153/153 bệnh, 0 metabolite bị loại thêm —
    tức patch này không thay đổi số liệu đã báo cáo, chỉ đồng bộ logic.

    COFACTORS là tham số BẮT BUỘC (không có default) để tránh vô tình
    quay lại hành vi bất đối xứng cũ nếu quên truyền.
    """
    from tqdm.auto import tqdm

    pw_file  = _get_smpdb_pw()
    met_dir  = _get_smpdb_met_dir()

    # Load pathways
    df_pw    = pd.read_csv(pw_file, low_memory=False)
    cols_low = {c.lower().strip(): c for c in df_pw.columns}
    smpid_col = next((cols_low[k] for k in cols_low
                      if 'smpdb' in k or k == 'smp id'), df_pw.columns[0])
    cat_col   = next((cols_low[k] for k in cols_low
                      if 'subject' in k or 'category' in k), None)
    name_col  = next((cols_low[k] for k in cols_low if 'name' in k),
                     df_pw.columns[2])

    if cat_col:
        dz_mask = df_pw[cat_col].astype(str).str.lower().str.strip() == 'disease'
        disease_pw_ids = set(df_pw.loc[dz_mask, smpid_col].astype(str))
    else:
        disease_pw_ids = set(df_pw[smpid_col].astype(str))

    smpid_to_name = dict(zip(df_pw[smpid_col].astype(str),
                             df_pw[name_col].astype(str)))
    print(f'  Disease pathways: {len(disease_pw_ids)}')

    # Load metabolites
    met_files = sorted(met_dir.glob('SMP*.csv'))
    if not met_files: met_files = sorted(met_dir.glob('*.csv'))

    # Build status cache for fast lookup
    hmdb_status_cache = {hid: m['status'] for hid, m in hmdb_metabolites.items()}

    def norm_hmdb_id(raw):
        if not raw or not isinstance(raw, str): return None
        raw = raw.strip()
        return standardize_hmdb_id(raw) if raw.upper().startswith('HMDB') else None

    smpdb_raw = defaultdict(set); hmdb_col = None; n_err = 0
    for mf in tqdm(met_files, desc='SMPDB files'):
        smp_id = mf.stem.split('_')[0]
        if smp_id not in disease_pw_ids: continue
        pw_name = smpid_to_name.get(smp_id, smp_id)
        if not pw_name or str(pw_name).lower() in ('nan','none',''): continue
        try: df_met = pd.read_csv(mf, low_memory=False)
        except Exception: n_err += 1; continue
        if hmdb_col is None:
            mc = {c.lower().strip(): c for c in df_met.columns}
            hmdb_col = next((mc[k] for k in mc if 'hmdb' in k), None)
        if hmdb_col is None or hmdb_col not in df_met.columns: continue
        for raw_id in df_met[hmdb_col]:
            hid = norm_hmdb_id(raw_id)
            if hid: smpdb_raw[pw_name].add(hid)
    print(f'  Loaded {len(smpdb_raw)} pathways (errors={n_err})')

    # Filter detected/quantified
    smpdb_filt = {}
    for pw_name, hmdb_ids in smpdb_raw.items():
        kept = {hid for hid in hmdb_ids
                if hmdb_status_cache.get(hid,'').strip().lower() in ALLOWED_STATUSES}
        if kept: smpdb_filt[pw_name] = kept

    # Map HMDB → Recon3D — lọc cả RECON3D_CURRENCY_METABOLITE (ID) và COFACTORS (tên),
    # đồng nhất với build_eval_set1/2 (patch 2026-07).
    smpdb_nodes = {}
    for pw_name, hmdb_ids in smpdb_filt.items():
        mnodes = set()
        for hid in hmdb_ids:
            base = hmdb_to_recon.get(hid)
            if not base or base not in node_idx: continue
            # Exact cofactor check, đồng nhất với build_eval_set1/2
            if base in RECON3D_CURRENCY_METABOLITE or normalize_name(
                    hmdb_metabolites.get(hid, {}).get('name', '')) in COFACTORS:
                continue
            mnodes.add(base)
        if mnodes: smpdb_nodes[pw_name] = mnodes

    # Filter None/empty keys
    smpdb_nodes = {k: v for k, v in smpdb_nodes.items()
                   if k and isinstance(k, str) and k.strip()
                   and str(k).lower() not in ('nan', 'none')}

    # ── Dedup 3 layers — exact từ notebook Cell 5 ──
    STOPWORDS = {'of','the','a','an','and','or','with','in','type','i','ii','iii',
                 'iv','v','1','2','3','disease','disorder','syndrome','deficiency'}

    def _norm(s):
        s = s.lower().strip()
        s = re.sub(r'\s*\([^)]*\)\s*$', '', s)
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        return re.sub(r'\s+', ' ', s).strip()

    def _sigtok(s):
        toks = set(_norm(s).split()) - STOPWORDS
        return toks if toks else set(_norm(s).split())

    def _jacc(a, b):
        ta, tb = _sigtok(a), _sigtok(b)
        return len(ta&tb)/len(ta|tb) if (ta and tb) else 0.0

    def _stk(s): return ' '.join(sorted(_sigtok(s)))

    # Layer 1: exact normalize
    grp1 = defaultdict(list)
    for dname in smpdb_nodes: grp1[_norm(dname)].append(dname)
    merged1 = {}
    for _, dnames in grp1.items():
        canon = max(dnames, key=len)
        merged1[canon] = set().union(*[smpdb_nodes[dn] for dn in dnames])

    # Layer 2: sorted-token key
    grp2 = defaultdict(list)
    for dname in merged1: grp2[_stk(dname)].append(dname)
    merged2 = {}
    for _, dnames in grp2.items():
        canon = max(dnames, key=len)
        merged2[canon] = set().union(*[merged1[dn] for dn in dnames])

    # Layer 3: Jaccard >= 0.85
    tok_idx = defaultdict(set)
    for dname in merged2:
        for tok in _sigtok(dname): tok_idx[tok].add(dname)
    cand_pairs = set()
    for tok, dnames in tok_idx.items():
        if len(dnames) < 2: continue
        for a, b in _combinations(sorted(dnames), 2): cand_pairs.add((a,b))
    alias_pairs = [(a,b) for a,b in cand_pairs if _jacc(a,b) >= 0.85]

    parent = {dname: dname for dname in merged2}
    def _find(x):
        while parent[x] != x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def _union(x, y):
        px, py = _find(x), _find(y)
        if px != py:
            if len(px) >= len(py): parent[py]=px
            else: parent[px]=py
    for a, b in alias_pairs: _union(a, b)

    grp3 = defaultdict(list)
    for dname in merged2: grp3[_find(dname)].append(dname)
    merged3 = {}
    for canon, aliases in grp3.items():
        merged3[canon] = set().union(*[merged2[a] for a in aliases])

    print(f'  Dedup: {len(smpdb_nodes)} → {len(merged1)} → {len(merged2)} → {len(merged3)}')

    eval_set3 = {
        dname: sorted(mnodes)
        for dname, mnodes in merged3.items()
        if len(mnodes) >= min_mets
    }
    return eval_set3

