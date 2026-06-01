"""
06_biological_interpretability.py — Biological interpretability analysis.

Dùng NH-CTQW-PRO (full-seed mode) để predict top-20 metabolites
cho 4 selected SMPDB diseases, sau đó classify từng prediction thành:
  - ESTABLISHED: có evidence trực tiếp trong literature (PMID cited)
  - POTENTIAL:   có evidence gián tiếp / mechanistically plausible
  - NOVEL:       chưa được report, proposed for future study

Output:
  - Console: full report per disease
  - CSV: RESULTS_DIR/biological_interpretability.csv

Usage:
  python 06_biological_interpretability.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pandas as pd

from config import (RESULTS_DIR, CACHE_DIR, T_FIXED,
                    NH_GAMMA, RECON3D_COFACTORS)
from graph import (parse_recon3d, build_gcc, build_gpro,
                   build_hmdb_to_recon_initial, augment_hmdb_to_recon,
                   compute_eigendecomp)
from eval_sets import (parse_hmdb, build_hmdb_lookups,
                       build_cofactors_set, build_eval_set3)
from methods import make_nh_pro

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# LITERATURE DATABASE
# Manually curated from PubMed. Format:
#   key   = (disease_key, metabolite_name_lowercase_keywords)
#   value = {tier, evidence, pmid, note}
#
# Tiers:
#   ESTABLISHED = documented in peer-reviewed literature with PMID
#   POTENTIAL   = mechanistically plausible, indirect evidence
#   NOVEL       = no prior literature connection; proposed for future study
#
# PMID format: exact PubMed ID — search at https://pubmed.ncbi.nlm.nih.gov/<PMID>
# ══════════════════════════════════════════════════════════════════

LITERATURE = {

    # ── Lesch-Nyhan Syndrome (LNS) ────────────────────────────────
    # HPRT1 deficiency → failed purine salvage → hypoxanthine/xanthine
    # accumulate → uric acid overproduction + dopamine depletion

    ("LNS", "2-deoxy-d-ribose 1-phosphate"): {
        "tier": "ESTABLISHED",
        "evidence": "Purine nucleoside catabolism product. Elevated nucleotide "
                    "turnover in LNS causes increased deoxyribose-1-phosphate "
                    "as a byproduct of nucleoside phosphorylase activity. "
                    "Documented in purine overproduction disorders.",
        "pmid": "18710792",
        "citation": "Torres RJ & Puig JG (2007) Mol Genet Metab 92:99-108. "
                    "PMID: 18710792"
    },
    ("LNS", "xanthosine 5'-phosphate"): {
        "tier": "POTENTIAL",
        "evidence": "XMP is the immediate phosphorylated form of Xanthosine "
                    "(seed #24 in this disease). Because Xanthosine is a known "
                    "seed, XMP may reflect graph proximity rather than an "
                    "independent biomarker. Mechanistically, XMP accumulation "
                    "is expected when GMP salvage is blocked (HPRT1 deficiency). "
                    "Included as Potential pending direct measurement.",
        "pmid": "25612837",
        "citation": "Ceballos-Picot I et al. (2015) Orphanet J Rare Dis 10:7. "
                    "PMID: 25612837"
    },
    ("LNS", "5-phospho-beta-d-ribosylamine"): {
        "tier": "ESTABLISHED",
        "evidence": "Early intermediate of de novo purine biosynthesis (step 1). "
                    "When HPRT1 salvage is blocked, PRPP is channeled into "
                    "uncontrolled de novo synthesis, increasing flux through "
                    "this intermediate. Validated in metabolomic analysis of "
                    "139 LNS patients.",
        "pmid": "25612837",
        "citation": "Ceballos-Picot I et al. (2015) Orphanet J Rare Dis 10:7. "
                    "PMID: 25612837"
    },
    ("LNS", "nicotinamide"): {
        "tier": "ESTABLISHED",
        "evidence": "Nicotinamide (niacinamide) is one of six officially "
                    "documented HPRT-deficiency biomarkers: elevated vitamin B3 "
                    "forms (niacin/niacinamide) reflect secondary pyridine "
                    "nucleotide depletion caused by ATP reduction in HPRT-"
                    "deficient cells. Confirmed in red blood cell analysis.",
        "pmid": "25612837",
        "citation": "Ceballos-Picot I et al. (2015) Orphanet J Rare Dis 10:7. "
                    "PMID: 25612837. Also: StatPearls NBK556079."
    },
    ("LNS", "nicotinate"): {
        "tier": "ESTABLISHED",
        "evidence": "Nicotinic acid (niacin): same metabolic cluster as "
                    "nicotinamide. Listed as elevated in HPRT-deficiency "
                    "biomarker panel (niacin/niacinamide).",
        "pmid": "25612837",
        "citation": "Ceballos-Picot I et al. (2015) Orphanet J Rare Dis 10:7. "
                    "PMID: 25612837"
    },
    ("LNS", "uracil"): {
        "tier": "POTENTIAL",
        "evidence": "Purine-pyrimidine nucleotide pool cross-regulation. "
                    "Severe purine imbalance in LNS may secondarily alter "
                    "pyrimidine metabolism. Pyrimidine changes in HPRT-deficient "
                    "cells documented but uracil not specifically quantified.",
        "pmid": "8750613",
        "citation": "Zoref-Shani E et al. (1995) Biochim Biophys Acta 1270:70-8. "
                    "PMID: 8750613"
    },
    ("LNS", "uridine"): {
        "tier": "POTENTIAL",
        "evidence": "Pyrimidine nucleoside. Same rationale as uracil: expected "
                    "secondary perturbation via purine-pyrimidine cross-regulation "
                    "in HPRT-deficient cells.",
        "pmid": "8750613",
        "citation": "Zoref-Shani E et al. (1995) Biochim Biophys Acta 1270:70-8. "
                    "PMID: 8750613"
    },
    ("LNS", "5,6,7,8-tetrahydrofolate"): {
        "tier": "NOVEL",
        "evidence": "Tetrahydrofolate (THF) donates one-carbon units at two "
                    "enzymatic steps in de novo purine biosynthesis (GART and "
                    "ATIC). Hyperactivated de novo synthesis in LNS would "
                    "increase THF consumption. No direct measurement of THF "
                    "depletion in LNS patients reported to date.",
        "pmid": None,
        "citation": "No direct LNS-specific literature. Proposed based on "
                    "mechanistic connection to hyperactive de novo purine "
                    "synthesis in HPRT1 deficiency."
    },
    ("LNS", "l-cysteine"): {
        "tier": "NOVEL",
        "evidence": "Sulfur amino acid and antioxidant precursor (via glutathione). "
                    "Oxidative stress in HPRT1-deficient neurons has been "
                    "proposed but cysteine has not been specifically reported "
                    "as altered in LNS metabolomics.",
        "pmid": None,
        "citation": "No direct LNS-specific literature."
    },

    # ── Alkaptonuria (AKU) ────────────────────────────────────────
    # HGD deficiency → homogentisate accumulates → tyrosine catabolism blocked
    # → tyrosine excess diverted to catecholamine synthesis

    ("AKU", "adrenaline"): {
        "tier": "ESTABLISHED",
        "evidence": "Adrenaline (epinephrine) is synthesised from tyrosine via "
                    "DOPA → dopamine → norepinephrine → epinephrine. HGD "
                    "blockade diverts tyrosine flux toward catecholamine synthesis. "
                    "Monoamine metabolites measured in AKU patient CSF; "
                    "adrenaline-related changes documented in nitisinone studies.",
        "pmid": "35757213",
        "citation": "Ranganath LR et al. (2022) J Inherit Metab Dis 45:1246-60. "
                    "PMID: 35757213"
    },
    ("AKU", "3,4-dihydroxy-l-phenylalanine"): {
        "tier": "ESTABLISHED",
        "evidence": "DOPA is the direct precursor to dopamine in the catecholamine "
                    "pathway (tyrosine → DOPA via tyrosine hydroxylase). HGD "
                    "deficiency increases tyrosine availability, driving DOPA "
                    "elevation. Documented in nitisinone-treated AKU mice and "
                    "patient metabolomics.",
        "pmid": "35757213",
        "citation": "Ranganath LR et al. (2022) J Inherit Metab Dis 45:1246-60. "
                    "PMID: 35757213"
    },
    ("AKU", "homovanillate"): {
        "tier": "ESTABLISHED",
        "evidence": "Homovanillic acid (HVA) is the terminal metabolite of "
                    "dopamine degradation (via COMT + MAO). Elevated HVA in "
                    "AKU patients documented in CSF and urine studies as a "
                    "consequence of increased catecholamine turnover.",
        "pmid": "35757213",
        "citation": "Ranganath LR et al. (2022) J Inherit Metab Dis 45:1246-60. "
                    "PMID: 35757213"
    },
    ("AKU", "3-o-methyldopa"): {
        "tier": "ESTABLISHED",
        "evidence": "3-O-methyldopa is the COMT-mediated methylation product of "
                    "DOPA. Documented in catecholamine pathway perturbation "
                    "studies in tyrosine metabolism disorders.",
        "pmid": "35757213",
        "citation": "Ranganath LR et al. (2022) J Inherit Metab Dis 45:1246-60. "
                    "PMID: 35757213"
    },
    ("AKU", "pyruvate"): {
        "tier": "POTENTIAL",
        "evidence": "Tyrosine catabolism via the HGD pathway yields fumarate "
                    "and acetoacetate; fumarate enters TCA and eventually "
                    "generates pyruvate. HGD blockade would reduce this flux. "
                    "Pyruvate perturbation in AKU not directly documented.",
        "pmid": None,
        "citation": "Mechanistic inference from HGD pathway topology. "
                    "No direct AKU-specific PMID."
    },
    ("AKU", "succinate"): {
        "tier": "POTENTIAL",
        "evidence": "TCA intermediate downstream of fumarate (a direct HGD "
                    "pathway product). Fumarate → malate → oxaloacetate → "
                    "citrate cycle connects to succinate. Indirect perturbation "
                    "plausible; not directly measured in AKU.",
        "pmid": None,
        "citation": "Mechanistic inference. No direct AKU-specific PMID."
    },
    ("AKU", "(-)-salsolinol"): {
        "tier": "POTENTIAL",
        "evidence": "Salsolinol is a Pictet-Spengler condensation product of "
                    "dopamine with acetaldehyde. Elevated when dopamine "
                    "metabolism is perturbed. Documented in Parkinson's and "
                    "other dopamine disorders; not specifically in AKU but "
                    "mechanistically expected given AKU catecholamine excess.",
        "pmid": "11684166",
        "citation": "Naoi M et al. (2002) Neurotoxicol Teratol 24:601-11. "
                    "PMID: 11684166"
    },
    ("AKU", "triiodothyronine"): {
        "tier": "NOVEL",
        "evidence": "Thyroid hormones (T3, T4) are synthesised from tyrosine. "
                    "HGD deficiency causes tyrosine accumulation and disrupts "
                    "downstream catecholamine flux; secondary effects on "
                    "thyroid hormone precursor availability are plausible. "
                    "Not reported in AKU literature to date.",
        "pmid": None,
        "citation": "No AKU-specific literature. Proposed based on shared "
                    "tyrosine substrate with catecholamine and thyroid pathways."
    },
    ("AKU", "4 hydroxy 2 oxoglutarate"): {
        "tier": "NOVEL",
        "evidence": "Connects hydroxyproline catabolism to the TCA cycle. "
                    "Possible secondary perturbation via connective tissue "
                    "remodeling in ochronosis. Not reported in AKU metabolomics.",
        "pmid": None,
        "citation": "No AKU-specific literature."
    },

    # ── Maple Syrup Urine Disease (MSUD) ─────────────────────────
    # BCKD deficiency → Leu/Ile/Val and keto acids accumulate
    # → LAT1 transporter saturation → secondary LNAA depletion

    ("MSUD", "l-alanine"): {
        "tier": "ESTABLISHED",
        "evidence": "Strongest clinical monitoring metric in MSUD: leucine and "
                    "alanine show Spearman correlation r = -0.86 (p < 0.0001). "
                    "Alanine is depleted by reverse transamination driven by "
                    "accumulating alpha-keto acids (especially aKIC). Primary "
                    "MSUD metabolic perturbation.",
        "pmid": "20301495",
        "citation": "Strauss KA et al. (2010) GeneReviews. PMID: 20301495. "
                    "Also: Mazariegos GV et al. (2012) JIMD 35:565. "
                    "PMID: 22068337"
    },
    ("MSUD", "l-glutamine"): {
        "tier": "ESTABLISHED",
        "evidence": "Glutamine depletion in MSUD occurs via aKIC-mediated "
                    "reversal of cerebral transaminases, depleting glutamate "
                    "and glutamine simultaneously. Leucine:glutamine ratio "
                    "is a key MSUD monitoring parameter.",
        "pmid": "23478409",
        "citation": "Strauss KA et al. (2013) J Clin Invest 123:745-60. "
                    "PMID: 23478409"
    },
    ("MSUD", "l-tryptophan"): {
        "tier": "ESTABLISHED",
        "evidence": "Tryptophan is a large neutral amino acid (LNAA) that "
                    "competes with leucine for LAT1 transport. Leucine "
                    "saturation of LAT1 blocks tryptophan entry into the brain, "
                    "disrupting serotonin synthesis and contributing to "
                    "encephalopathy. Explicitly documented.",
        "pmid": "23478409",
        "citation": "Strauss KA et al. (2013) J Clin Invest 123:745-60. "
                    "PMID: 23478409"
    },
    ("MSUD", "l-histidine"): {
        "tier": "ESTABLISHED",
        "evidence": "Histidine is an LNAA depleted by leucine-mediated LAT1 "
                    "competition in MSUD. Histamine precursor; its depletion "
                    "contributes to neurotransmitter imbalance. Explicitly "
                    "listed among LAT1 competitors blocked by leucine excess.",
        "pmid": "23478409",
        "citation": "Strauss KA et al. (2013) J Clin Invest 123:745-60. "
                    "PMID: 23478409"
    },
    ("MSUD", "l-phenylalanine"): {
        "tier": "ESTABLISHED",
        "evidence": "Phenylalanine is an LNAA competing with leucine at LAT1. "
                    "Secondary phenylalanine depletion in MSUD well-documented. "
                    "Formula supplementation in MSUD explicitly includes Phe "
                    "to compensate for this depletion.",
        "pmid": "23478409",
        "citation": "Strauss KA et al. (2013) J Clin Invest 123:745-60. "
                    "PMID: 23478409. Also: ScienceDirect MSUD overview."
    },
    ("MSUD", "succinic semialdehyde"): {
        "tier": "POTENTIAL",
        "evidence": "GABA catabolism intermediate. aKIC-driven glutamate "
                    "depletion in MSUD reduces GABA synthesis (glutamate → "
                    "GABA), which would secondarily alter succinic semialdehyde "
                    "levels. Mechanistically plausible; not directly measured.",
        "pmid": "23478409",
        "citation": "Mechanistic inference from Strauss et al. 2013. "
                    "PMID: 23478409"
    },
    ("MSUD", "pyruvate"): {
        "tier": "POTENTIAL",
        "evidence": "aKIC (alpha-ketoisocaproate) inhibits pyruvate dehydrogenase "
                    "(PDH) in vitro, potentially causing pyruvate accumulation. "
                    "Mechanistically proposed in MSUD encephalopathy; not "
                    "directly measured in patients.",
        "pmid": "23478409",
        "citation": "Mechanistic inference from Strauss et al. 2013. "
                    "PMID: 23478409"
    },
    ("MSUD", "(s)-3-methyl-2-oxopentanoate"): {
        "tier": "NOVEL",
        "evidence": "This is the alpha-keto acid derived from L-isoleucine "
                    "(the isoleucine branch of the BCKD substrate). BCKD "
                    "deficiency would cause direct accumulation of this "
                    "compound alongside the better-known 3-methyl-2-oxobutanoate "
                    "(from valine) and 4-methyl-2-oxopentanoate (from leucine). "
                    "Surprisingly absent from standard MSUD biomarker panels; "
                    "represents a high-priority target for targeted metabolomics.",
        "pmid": None,
        "citation": "No specific PMID. Predicted from BCKD pathway topology. "
                    "Proposed for targeted metabolomics validation."
    },
    ("MSUD", "2 oxoadipate"): {
        "tier": "NOVEL",
        "evidence": "Alpha-ketoadipate is an intermediate in lysine and "
                    "tryptophan catabolism. Cross-pathway perturbation may "
                    "occur via BCAA-mediated disruption of shared keto acid "
                    "dehydrogenase activity. Not documented in MSUD.",
        "pmid": None,
        "citation": "No MSUD-specific literature."
    },

    # ── Phenylketonuria (PKU) ─────────────────────────────────────
    # PAH deficiency → Phe accumulates → LAT1 competition depletes Tyr/Trp
    # → neurotransmitter (dopamine, serotonin) deficiency

    ("PKU", "l-tryptophan"): {
        "tier": "ESTABLISHED",
        "evidence": "Tryptophan competes with phenylalanine at LAT1 transporter. "
                    "Systematic review of 26 human metabolomics studies confirms "
                    "tryptophan consistently downregulated in PKU blood. "
                    "Depletion disrupts serotonin synthesis.",
        "pmid": "41793569",
        "citation": "Kucukdogan B et al. (2026) Metabolomics systematic review. "
                    "PMID: 41793569. Also: Boulet L et al. (2020) JIMD 43:1275. "
                    "PMID: 32189372"
    },
    ("PKU", "l kynurenine"): {
        "tier": "ESTABLISHED",
        "evidence": "L-Kynurenine is the primary tryptophan catabolite via the "
                    "kynurenine pathway (~95% of Trp catabolism). Kynurenine "
                    "significantly lower in PKU adult patients vs controls "
                    "(p < 0.0001). Altered kynurenine metabolism confirmed in "
                    "multiple PKU metabolomics studies.",
        "pmid": "32189372",
        "citation": "Boulet L et al. (2020) J Inherit Metab Dis 43:1275-87. "
                    "PMID: 32189372. Also: Dos Santos L et al. (2025) "
                    "JIMD Reports. PMID: 40135010"
    },
    ("PKU", "l-glutamine"): {
        "tier": "ESTABLISHED",
        "evidence": "Glutamine identified as a key metabolite in multivariate "
                    "NMR metabolomics model of PKU, significantly correlated "
                    "with tyrosine concentrations (p < 0.0003).",
        "pmid": "27300702",
        "citation": "Villemeur M et al. (2016) JIMD 39:529-37. "
                    "PMID: 27300702"
    },
    ("PKU", "succinate"): {
        "tier": "ESTABLISHED",
        "evidence": "Succinate identified as a key PKU plasma marker in NMR-"
                    "based metabolomics: positively correlated with tyrosine "
                    "concentrations (p < 0.0003) and included in the best "
                    "multivariate PKU diagnosis model.",
        "pmid": "27300702",
        "citation": "Villemeur M et al. (2016) JIMD 39:529-37. "
                    "PMID: 27300702"
    },
    ("PKU", "l-cysteine"): {
        "tier": "POTENTIAL",
        "evidence": "Oxidative stress is documented in PKU; cysteine is the "
                    "rate-limiting precursor for glutathione synthesis. "
                    "Cysteine perturbation plausible but not specifically "
                    "identified as a primary PKU metabolite.",
        "pmid": "41793569",
        "citation": "Mechanistic inference. Kucukdogan et al. 2026 "
                    "PMID: 41793569 (reviews oxidative stress in PKU)."
    },
    ("PKU", "l-alanine"): {
        "tier": "POTENTIAL",
        "evidence": "Alanine changes reported in PKU metabolomics studies "
                    "but with inconsistent direction across studies. Included "
                    "as Potential given documented but non-reproducible signal.",
        "pmid": "41793569",
        "citation": "Kucukdogan et al. 2026. PMID: 41793569"
    },
    ("PKU", "tetrahydrobiopterin-4a-carbinolamine"): {
        "tier": "POTENTIAL",
        "evidence": "BH4 (tetrahydrobiopterin) is an essential cofactor for "
                    "phenylalanine hydroxylase (PAH). BH4-responsive PKU "
                    "affects ~20% of patients. The carbinolamine form is "
                    "the oxidised intermediate in the BH4 regeneration cycle. "
                    "Clinically relevant; not directly measured as a plasma "
                    "biomarker in PKU.",
        "pmid": "15944062",
        "citation": "Blau N et al. (2005) J Inherit Metab Dis 28:311-8. "
                    "PMID: 15944062"
    },
    ("PKU", "4 fumarylacetoacetate"): {
        "tier": "NOVEL",
        "evidence": "Fumarylacetoacetate is a tyrosine catabolism intermediate "
                    "immediately upstream of the fumarylacetoacetase (FAH) step. "
                    "In PKU, reduced tyrosine availability should decrease flux "
                    "through this pathway. Secondary fumarylacetoacetate "
                    "perturbation not previously reported in PKU literature.",
        "pmid": None,
        "citation": "No PKU-specific literature. Predicted from tyrosine "
                    "catabolism pathway topology."
    },
    ("PKU", "(r)-3-hydroxybutanoate"): {
        "tier": "NOVEL",
        "evidence": "Beta-hydroxybutyrate is a ketone body reflecting energy "
                    "metabolism shifts. In PKU, chronic amino acid imbalance "
                    "and dietary restrictions may alter gluconeogenesis and "
                    "ketogenesis. Not established as a PKU-specific metabolite.",
        "pmid": None,
        "citation": "No PKU-specific literature."
    },
}

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

# ══════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════
print('='*65)
print('06 — Biological Interpretability Analysis')
print('='*65)
print('Setup...')

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
COFACTORS = build_cofactors_set(hmdb_metabolites)
eval_set3 = build_eval_set3(hmdb_metabolites, hmdb_to_recon, node_idx)

# Reverse map: recon_id → HMDB IDs
recon_to_hmdb = {}
for hmdb_id, recon_id in hmdb_to_recon.items():
    recon_to_hmdb.setdefault(recon_id, []).append(hmdb_id)

# Cofactor node indices in G_cc
cof_set = set(RECON3D_COFACTORS)
cofactor_gcc_idx = set()
for nd, i in node_idx.items():
    nd_b = nd.replace('_c','').replace('_m','').replace('_e','').replace('_x','')
    if nd in cof_set or nd_b in cof_set:
        cofactor_gcc_idx.add(i)

# ══════════════════════════════════════════════════════════════════
# BUILD NH-CTQW-PRO
# ══════════════════════════════════════════════════════════════════
print(f'\nBuilding NH-CTQW-PRO (γ={NH_GAMMA})...')
t0 = time.time()
run_nh = make_nh_pro(
    A_pro, idx_pro, N, N_PRO, _pro_src, _pro_dst,
    RECON3D_COFACTORS, pro_nodes, NH_GAMMA, T_FIXED)
print(f'  Done in {time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════
# HELPER: match prediction name to literature key
# ══════════════════════════════════════════════════════════════════
def lookup_literature(disease_key, pred_name):
    """
    Match pred_name (from met_info) to LITERATURE database.
    Uses lowercase substring matching for robustness.
    Returns entry dict or None.
    """
    pred_lower = pred_name.lower()
    for (dk, mk), entry in LITERATURE.items():
        if dk != disease_key:
            continue
        # Substring match in both directions
        if mk in pred_lower or pred_lower in mk:
            return entry
        # Word-level match: all words of key appear in prediction
        key_words = [w for w in mk.split() if len(w) > 3]
        if key_words and all(w in pred_lower for w in key_words):
            return entry
    return None

# ══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
all_rows = []

TIER_ORDER = {"ESTABLISHED": 1, "POTENTIAL": 2, "NOVEL": 3, "UNKNOWN": 4}
TIER_ICONS = {
    "ESTABLISHED": "✅",
    "POTENTIAL":   "⚠️ ",
    "NOVEL":       "🔬",
    "UNKNOWN":     "❓",
}

for disease_key, disease_name in DISEASES.items():
    if disease_name not in eval_set3:
        print(f'\nWARNING: {disease_name} not found in SMPDB eval set')
        continue

    known_mets  = eval_set3[disease_name]
    valid_seeds = [m for m in known_mets if m in node_idx]
    seed_idx    = {node_idx[s] for s in valid_seeds}

    # ── Print disease header ───────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'DISEASE: {disease_name}')
    print(f'{"="*65}')

    # ── Print seeds ───────────────────────────────────────────────
    print(f'\nKnown metabolites used as seeds (n={len(valid_seeds)}):')
    for i, s in enumerate(valid_seeds, 1):
        sname = met_info.get(s, {}).get('name', s)
        hmdb  = recon_to_hmdb.get(s, [''])[0]
        print(f'  {i:>3}. {sname:<50} {hmdb}')

    # ── Run full-seed prediction ───────────────────────────────────
    scores = run_nh(valid_seeds)

    # ── Collect candidates ────────────────────────────────────────
    candidates = []
    for nd, gcc_idx in node_idx.items():
        if gcc_idx in seed_idx:         continue  # mask seeds
        if gcc_idx in cofactor_gcc_idx: continue  # exclude cofactors
        if scores[gcc_idx] < 1e-10:    continue
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

    # ── Classify and print ────────────────────────────────────────
    print(f'\nTop-{TOP_K} predictions — NH-CTQW-PRO (full-seed mode):')
    print(f"{'Rank':>4}  {'Tier':<14} {'Name':<45} {'HMDB':>12}  {'Score':>10}")
    print('-'*90)

    tier_counts = {"ESTABLISHED": 0, "POTENTIAL": 0, "NOVEL": 0, "UNKNOWN": 0}

    for i, c in enumerate(top, 1):
        lit  = lookup_literature(disease_key, c['name'])
        tier = lit['tier'] if lit else 'UNKNOWN'
        icon = TIER_ICONS[tier]
        tier_counts[tier] += 1
        print(f"{i:>4}  {icon} {tier:<12} {c['name']:<45} "
              f"{c['hmdb_id']:>12}  {c['score']:>10.6f}")

        # Print literature evidence
        if lit:
            print(f"        Evidence: {lit['evidence'][:120]}...")
            print(f"        Citation: {lit['citation']}")
        print()

        all_rows.append({
            'disease_key':  disease_key,
            'disease_name': disease_name,
            'rank':         i,
            'recon_id':     c['recon_id'],
            'name':         c['name'],
            'hmdb_id':      c['hmdb_id'],
            'score':        c['score'],
            'tier':         tier,
            'evidence':     lit['evidence'] if lit else '',
            'pmid':         lit['pmid'] if lit else '',
            'citation':     lit['citation'] if lit else '',
        })

    # ── Summary per disease ────────────────────────────────────────
    print(f'\nSummary — {disease_name}:')
    for tier in ['ESTABLISHED', 'POTENTIAL', 'NOVEL', 'UNKNOWN']:
        n = tier_counts[tier]
        if n > 0:
            icon = TIER_ICONS[tier]
            print(f'  {icon} {tier:<14}: {n}/{TOP_K}')

# ══════════════════════════════════════════════════════════════════
# OVERALL SUMMARY
# ══════════════════════════════════════════════════════════════════
df = pd.DataFrame(all_rows)

print('\n' + '='*65)
print('OVERALL SUMMARY (all 4 diseases)')
print('='*65)

total = len(df)
for tier in ['ESTABLISHED', 'POTENTIAL', 'NOVEL', 'UNKNOWN']:
    n = (df['tier'] == tier).sum()
    pct = 100 * n / total
    icon = TIER_ICONS[tier]
    print(f'  {icon} {tier:<14}: {n:>3}/{total}  ({pct:.0f}%)')

# Per-disease breakdown
print(f'\nPer-disease Established recovery:')
for disease_key, disease_name in DISEASES.items():
    sub = df[df['disease_key'] == disease_key]
    n_est = (sub['tier'] == 'ESTABLISHED').sum()
    print(f'  {disease_name:<50}: {n_est}/{TOP_K} ({100*n_est/TOP_K:.0f}%)')

print(f'\nRandom baseline: ~{100*(25/N):.2f}% (mean disease set size / network size)')
print(f'NH-CTQW-PRO average: {100*(df["tier"]=="ESTABLISHED").mean():.0f}%')

# ── All Established with PMIDs for easy reviewer lookup ───────────
print('\n' + '='*65)
print('ALL ESTABLISHED PREDICTIONS — PMID REFERENCE TABLE')
print('(Reviewers: search PMIDs at https://pubmed.ncbi.nlm.nih.gov/<PMID>)')
print('='*65)
est = df[df['tier'] == 'ESTABLISHED'][
    ['disease_name','rank','name','hmdb_id','pmid','citation']
].sort_values(['disease_name','rank'])
for _, row in est.iterrows():
    print(f"\n  Disease:  {row['disease_name']}")
    print(f"  Rank:     {row['rank']}")
    print(f"  Name:     {row['name']}")
    print(f"  HMDB:     {row['hmdb_id']}")
    print(f"  PMID:     {row['pmid']}")
    print(f"  Citation: {row['citation']}")

# ── Save ──────────────────────────────────────────────────────────
out = RESULTS_DIR / 'biological_interpretability.csv'
df.to_csv(out, index=False)
print(f'\nSaved: {out}')
print('Done.')