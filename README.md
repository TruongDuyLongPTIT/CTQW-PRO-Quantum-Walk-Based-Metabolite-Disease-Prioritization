# CTQW-PRO: Quantum Walk-Based Metabolite–Disease Prioritization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TruongDuyLongPTIT/CTQW_PRO_METABOLITES_PRIORITIZING/blob/main/main_notebook.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Continuous-Time Quantum Walk on metabolic networks for disease-metabolite prioritization — no training required.**

## Overview

We compare three graph-based ranking methods on the **Recon3D** human metabolic network:

| Method | Description |
|--------|-------------|
| **PROFANCY** | Random Walk with Restart (classical baseline) |
| **CTQW-PRO** | Continuous-Time Quantum Walk on bipartite metabolite–pathway graph |
| **Driven CTQW-PRO** | CTQW with iterative seed reinforcement |

Evaluation uses **leave-one-out cross-validation** on three independent disease-metabolite sets (HMDB+CTD, MarkerDB, SMPDB), measuring AUC, MRR, and Recall@k.

## Key Results

- CTQW-PRO consistently outperforms PROFANCY across all three evaluation sets
- Driven CTQW-PRO further improves MRR by ~22% on SMPDB
- CTQW-PRO advantage is largest for diseases with dispersed metabolites (Spearman r = +0.53, p < 0.001)
- Optimal time parameter t ≈ 0.09–0.10 on Recon3D

## Quickstart

Click **Open in Colab** above, then run **Cell 0** to clone the repo and install dependencies. Data files must be available in Google Drive (see below).

## Data

The following files are required in Google Drive under `MyDrive/CTQW for metabolites/`:

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
├── main_notebook.ipynb       # Main notebook (run this)
├── src/
│   ├── config.py             # Paths & hyperparameters
│   ├── graph.py              # Recon3D parsing, G_pro construction
│   ├── methods.py            # PROFANCY, CTQW-PRO, Driven CTQW-PRO
│   ├── evaluation.py         # LOO eval, metrics
│   ├── eval_sets.py          # HMDB+CTD, MarkerDB, SMPDB loaders
│   └── utils.py
└── experiments/
    ├── 01_main_results.py    # Main LOO evaluation
    ├── 02_ablation_graph.py  # Graph ablation (cofactor filtering, t-sweep)
    ├── 03_negative_results.py # Chiral walk, geometric t, self-loop leakage
    └── 04_figures.py         # Publication figures
```

## Requirements

```
torch >= 2.0
scikit-learn
networkx
numpy
pandas
scipy
tqdm
```

Install via:
```bash
pip install torch scikit-learn networkx numpy pandas scipy tqdm
```

## Methods

**CTQW-PRO** evolves an initial quantum state $|\psi_0\rangle$ (uniform superposition of seed metabolites) over a bipartite graph $G_\text{pro}$ combining the Recon3D metabolic network with pathway nodes:

$$|\psi(t)\rangle = e^{-iA_\text{pro} t}|\psi_0\rangle$$

Metabolite scores are the diagonal probabilities $|\langle v|\psi(t)\rangle|^2$ at fixed $t = 0.1$.

**Driven CTQW-PRO** periodically reinforces the walker toward seed nodes:

$$|\psi_{k+1}\rangle \propto (1-\alpha)\, e^{-iA_\text{pro}\delta t}|\psi_k\rangle + \alpha\,|\psi_0\rangle$$

## Citation

```bibtex
@article{ctqwpro2025,
  title   = {CTQW-PRO: Quantum Walk-Based Metabolite–Disease Prioritization},
  author  = {...},
  journal = {...},
  year    = {202x}
}
```

## Related Work

- Saarinen et al. (2024). [Disease gene prioritization with quantum walks.](https://academic.oup.com/bioinformatics/article/40/8/btae513/7738783) *Bioinformatics.*
- Dubovitskii et al. (2025). [On Quantum Random Walks in Biomolecular Networks.](https://arxiv.org/abs/2506.06514) *arXiv.*
