"""
config.py — Single source of truth. Exact từ notebook Cell 1.
"""
import os
from pathlib import Path

BASE_DIR    = Path('/content/drive/MyDrive/CTQW for metabolites')
RESULTS_DIR = BASE_DIR / 'results'
CACHE_DIR   = RESULTS_DIR / 'cache'

PATH_RECON3D  = BASE_DIR / 'Recon3D.json'
PATH_HMDB_ZIP = BASE_DIR / 'hmdb_metabolites.zip'
PATH_CTD      = BASE_DIR / 'CTD_chemicals_diseases.csv.gz'
PATH_MARKERDB = BASE_DIR / 'all_chemicals.xml'
PATH_SMPDB_PW = BASE_DIR / 'smpdb_pathways.csv.zip'
PATH_SMPDB_MET= BASE_DIR / 'smpdb_metabolites.csv.zip'
SMPDB_MET_DIR = BASE_DIR / 'smpdb_metabolites.csv'
SMPDB_PW_DIR  = BASE_DIR / 'smpdb_pathways.csv'

T_FIXED     = 0.1
NH_GAMMA    = 22.0   # mean_degree G_pro — từ grid search
RRF_K       = 60     # Cormack et al. 2009 default
RWR_R   = 0.85
RWR_TOL     = 1e-8
RWR_MAXITER = 200
DRIVEN_N_STEPS = 2
DRIVEN_ALPHA   = 0.5

MIN_METS    = 8      # notebook output = 8
RANDOM_SEED = 42
METRIC_KEYS_FULL = ['auc', 'mrr', 'rank', 'r@5', 'r@10', 'r@20', 'r@50']

# Exact từ notebook Cell 3
RECON3D_CURRENCY_METABOLITE = {
    'h2o','h','co2','o2','pi','ppi','hco3','atp','adp','amp','gtp','gdp','gmp',
    'ctp','cdp','cmp','utp','udp','ump','datp','dadp','damp','nad','nadh','nadp',
    'nadph','fad','fadh2','fmn','fmnh2','coa','accoa','q','qh2','h2o2',
    'na1','k1','cl','ca2','mg2','fe2','fe3','zn2','cu2','mn2','nh3','nh4',
    'so4','no','h2','oh1','h2s',
}
CURRENCY_METABOLITE_FALLBACK = {
    'h2o','h','h+','oh-','na+','k+','cl-','ca2+','mg2+','fe2+','fe3+',
    'atp','adp','amp','gtp','gdp','gmp','ctp','cdp','cmp',
    'utp','udp','ump','ttp','tdp','tmp',
    'nad+','nadh','nadp+','nadph','fad','fadh2','fmn','fmnh2',
    'coa','acetyl-coa','co2','o2','hco3-','h2o2','pi','ppi',
    'so4','no','nh3','nh4+',
}
GENERIC_DISEASES = {
    'neoplasms','inflammation','disease','syndrome','death','pain',
    'fever','fatigue','tumor','tumors','carcinoma','cancer','disorders',
    'disease_models_animal','general_pathological_conditions','animal_diseases',
    'cell_death','genetic_diseases_inborn',
}
ALLOWED_STATUSES = {'detected', 'quantified', 'detected and quantified'}

N_JOBS = os.cpu_count()
for _env in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(_env, str(N_JOBS))


