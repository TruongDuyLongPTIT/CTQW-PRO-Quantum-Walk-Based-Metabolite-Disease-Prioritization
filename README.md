# CTQW-PRO: Quantum Walk-Based Metabolite–Disease Prioritization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TruongDuyLongPTIT/CTQW_PRO_METABOLITES_PRIORITIZING/blob/main/main_notebook.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Continuous-Time Quantum Walk on metabolic networks for disease-metabolite prioritization — no training required.**

## Overview

We compare three graph-based ranking methods on the **Recon3D** human metabolic network (2,788 nodes, 22,439 edges):

| Method | Description |
|--------|-------------|
| **PROFANCY** | Random Walk with Restart on bipartite graph G_pro |
| **CTQW-PRO** | Continuous-Time Quantum Walk on bipartite metabolite–pathway graph G_pro |
| **Driven CTQW-PRO** | CTQW-PRO with iterative seed reinforcement (α=0.5, 2 steps) |

Evaluation uses **leave-one-out cross-validation** across three independent disease-metabolite sets, all results reported as Wilcoxon paired test (Bonferroni-corrected).

## Key Results

### Table 1 — CTQW vs RWR on G_cc (graph without pathway nodes)

| Set | n | Method | AUC | MRR | R@5 | R@10 | R@20 |
|-----|---|--------|-----|-----|-----|------|------|
| HMDB+CTD | 158 | RWR | 0.769 | 0.011 | 0.000 | 0.001 | 0.045 |
| | | **CTQW** | 0.762 | **0.051** | **0.077** | **0.118** | **0.177** |
| MarkerDB | 21 | RWR | 0.835 | 0.018 | 0.009 | 0.009 | 0.094 |
| | | **CTQW** | **0.841** | **0.076** | **0.106** | **0.170** | **0.245** |
| SMPDB | 153 | RWR | **0.934** | 0.027 | 0.000 | 0.016 | 0.171 |
| | | **CTQW** | 0.931 | **0.193** | **0.274** | **0.351** | **0.479** |

### Table 2 — CTQW-PRO vs PROFANCY on G_pro (with pathway nodes)

| Set | n | Method | AUC | MRR | R@5 | R@10 | R@20 |
|-----|---|--------|-----|-----|-----|------|------|
| HMDB+CTD | 158 | PROFANCY | 0.810 | 0.011 | 0.000 | 0.001 | 0.045 |
| | | **CTQW-PRO** | **0.820**\*\* | **0.060**\*\*\* | **0.085**\*\*\* | **0.134**\*\*\* | **0.192**\*\*\* |
| MarkerDB | 21 | PROFANCY | 0.864 | 0.018 | 0.005 | 0.009 | 0.090 |
| | | **CTQW-PRO** | **0.884** | **0.085**\*\*\* | **0.130**\*\* | **0.186**\*\* | **0.265**\* |
| SMPDB | 153 | PROFANCY | 0.945 | 0.027 | 0.000 | 0.014 | 0.179 |
| | | **CTQW-PRO** | **0.957**\*\*\* | **0.217**\*\*\* | **0.308**\*\*\* | **0.418**\*\*\* | **0.543**\*\*\* |

> \*p<0.05 · \*\*p<0.01 · \*\*\*p<0.001 (Wilcoxon paired, Bonferroni-corrected)

**Win rate (AUC, CTQW-PRO vs PROFANCY):** 63% on HMDB+CTD · 62% on MarkerDB · **80% on SMPDB**

### Table 3 — Driven CTQW-PRO vs CTQW-PRO on G_pro

Driven variant tested on SMPDB only (largest set):

| Method | AUC | MRR | R@5 | R@10 | R@20 | R@50 |
|--------|-----|-----|-----|------|------|------|
| CTQW-PRO | **0.957** | 0.217 | 0.308 | 0.418 | 0.543 | 0.674 |
| **Driven CTQW-PRO** | 0.955 | **0.266**\*\*\* | **0.391**\*\*\* | **0.489**\*\*\* | **0.593**\*\*\* | **0.735**\*\*\* |

> Driven CTQW-PRO improves MRR by **+22.5%**, R@5 by **+26.9%**, R@20 by **+9.2%** over CTQW-PRO (all p<0.001).

### Bootstrap 95% Confidence Intervals (key metrics)

| Set | Method | AUC | MRR | R@20 |
|-----|--------|-----|-----|------|
| HMDB+CTD | PROFANCY | 0.810 [0.797, 0.824] | 0.011 [0.010, 0.013] | 0.045 [0.032, 0.059] |
| | CTQW-PRO | 0.820 [0.804, 0.836] | 0.060 [0.046, 0.074] | 0.192 [0.155, 0.227] |
| MarkerDB | PROFANCY | 0.864 [0.832, 0.900] | 0.018 [0.014, 0.023] | 0.090 [0.040, 0.154] |
| | CTQW-PRO | 0.884 [0.857, 0.910] | 0.085 [0.057, 0.123] | 0.265 [0.193, 0.340] |
| SMPDB | PROFANCY | 0.945 [0.942, 0.949] | 0.027 [0.026, 0.029] | 0.179 [0.157, 0.200] |
| | CTQW-PRO | 0.957 [0.952, 0.961] | 0.217 [0.197, 0.238] | 0.543 [0.516, 0.571] |

### Additional findings
- CTQW-PRO advantage over PROFANCY grows with metabolite dispersion (Spearman r = +0.53, p < 0.001)
- On high-dispersion diseases: PROFANCY R@20 ≈ 0.007 vs CTQW-PRO R@20 = 0.115 (**16× improvement**)
- Optimal time parameter t ≈ 0.09–0.10; high-dispersion diseases benefit from t ≈ 0.05
- Pathway-augmented graph G_pro substantially outperforms plain G_cc for all methods

## Quickstart

Click **Open in Colab** above, then run **Cell 0** to clone the repo and install dependencies. Data files must be available in Google Drive (see below).

## Data

Place the following files in Google Drive under `MyDrive/CTQW for metabolites/`:

| File | Source |
|------|--------|
| `Recon3D.json` | [BiGG Models](http://bigg.ucsd.edu/models/Recon3D) |
| `hmdb_metabolites.zip` | [HMDB](https://hmdb.ca/downloads) |
| `CTD_chemicals_diseases.csv.gz` | [CTD](http://ctdbase.org/downloads/) |
| `all_chemicals.xml` | [MarkerDB](https://markerdb.ca/downloads) |
| `smpdb_pathways.csv.zip` | [SMPDB](https://smpdb.ca/downloads) |
| `smpdb_metabolites.csv.zip` | [SMPDB](https://smpdb.ca/downloads) |

## Project Structure

```
├── main_notebook.ipynb         # Main notebook (run this)
├── src/
│   ├── config.py               # Paths & hyperparameters
│   ├── graph.py                # Recon3D parsing, G_pro construction
│   ├── methods.py              # PROFANCY, CTQW-PRO, Driven CTQW-PRO
│   ├── evaluation.py           # LOO eval, metrics
│   ├── eval_sets.py            # HMDB+CTD, MarkerDB, SMPDB loaders
│   └── utils.py
└── experiments/
    ├── 01_main_results.py      # Main LOO evaluation (Tables 1–3 + Wilcoxon)
    ├── 02_ablation_graph.py    # Graph ablation (cofactor filtering, t-sweep)
    ├── 03_negative_results.py  # Chiral walk, geometric t, self-loop leakage
    └── 04_figures.py           # Publication figures
```

## Requirements

```bash
pip install torch scikit-learn networkx numpy pandas scipy tqdm
```

## Methods

**CTQW-PRO** evolves an initial quantum state $|\psi_0\rangle$ (uniform superposition of seed metabolites) over a bipartite graph $G_\text{pro}$ combining Recon3D with 106 pathway nodes:

$$|\psi(t)\rangle = e^{-iA_\text{pro}\, t}|\psi_0\rangle, \quad t = 0.1$$

Metabolite scores are the measurement probabilities $|\langle v|\psi(t)\rangle|^2$.

**Driven CTQW-PRO** periodically reinforces the walker toward seed nodes:

$$|\psi_{k+1}\rangle \propto (1-\alpha)\,e^{-iA_\text{pro}\,\delta t}|\psi_k\rangle + \alpha\,|\psi_0\rangle, \quad \alpha=0.5,\ \delta t=0.1$$

## Citation

```bibtex
@article{ctqwpro2025,
  title   = {CTQW-PRO: Quantum Walk-Based Metabolite–Disease Prioritization},
  author  = {...},
  journal = {...},
  year    = {2025}
}
```

## Related Work

- Saarinen et al. (2024). [Disease gene prioritization with quantum walks.](https://academic.oup.com/bioinformatics/article/40/8/btae513/7738783) *Bioinformatics.*
- Dubovitskii et al. (2025). [On Quantum Random Walks in Biomolecular Networks.](https://arxiv.org/abs/2506.06514) *arXiv.*
