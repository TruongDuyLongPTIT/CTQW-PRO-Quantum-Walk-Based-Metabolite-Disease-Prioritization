# CTQW-PRO: Quantum Walk-Based Metabolite–Disease Prioritization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TruongDuyLongPTIT/CTQW-PRO-Quantum-Walk-Based-Metabolite-Disease-Prioritization/blob/main/main_notebook_recon3d.ipynb) Recon3D
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TruongDuyLongPTIT/CTQW-PRO-Quantum-Walk-Based-Metabolite-Disease-Prioritization/blob/main/main_notebook_kegg.ipynb) KEGG
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code accompanying our manuscript on CTQW-PRO (under review). It reproduces all experiments reported in the manuscript.

## Overview

Given a set of metabolites known to be associated with a disease (seeds), each method scores all other metabolites in a metabolic network; candidates are ranked by score. No training is required.

**Methods proposed in this work**

| Method | Graph | Description |
|---|---|---|
| CTQW | $G_{cc}$ | Continuous-time quantum walk with Hamiltonian $H = A$ |
| CTQW-PRO | $G_{pro}$ | CTQW on the pathway-augmented graph |
| NH-CTQW-PRO | $G_{pro}$ | CTQW-PRO with a non-Hermitian term that attenuates amplitude at currency metabolites |

**Baselines**

| Method | Reference |
|---|---|
| RWR | Köhler et al., 2008 |
| PROFANCY (RWR on $G_{pro}$) | Shang et al., 2014 |
| DADA-EC | Erten et al., 2011 |
| NetCore (core, diff, ratio variants) | Barel & Herwig, 2020 |
| MetaboRank-LITE | reduced version of MetaboRank (Frainay et al., 2019) |
| AMEND-IN-LITE | Inflation-Normalization subroutine of AMEND 2.0 (Boyd et al., 2025) |

## Method summary

**Graphs.** $G_{cc}$ is the largest connected component of the metabolite graph, in which two metabolites are linked if they occur in the same reaction. $G_{pro}$ extends $G_{cc}$ with one node per pathway, linked to all metabolites of that pathway.

**CTQW / CTQW-PRO.** The initial state is a uniform superposition over the seed metabolites, $|\psi_0\rangle$. The state evolves as

$$|\psi(t)\rangle = e^{-iAt}\,|\psi_0\rangle, \qquad t = 0.1,$$

and the score of metabolite $v$ is $|\langle v|\psi(t)\rangle|^2$.

**NH-CTQW-PRO.** The Hamiltonian on $G_{pro}$ is replaced by

$$H_{\mathrm{eff}} = A_{pro} - i\gamma\,\mathrm{diag}(c),$$

where $c_v = 1$ if $v$ is one of the 29 currency metabolites listed in `src/config.py` and $c_v = 0$ otherwise. $\gamma$ is set to the mean degree of $G_{pro}$ (22 for Recon3D, 16 for KEGG). Scores are computed as for CTQW-PRO.

**Hyperparameters.** $t = 0.1$ was selected by a grid search for CTQW-PRO on Recon3D / HMDB+CTD (`05_supplementary.py`, S2) and used unchanged for all networks and disease sets. RWR-based methods use restart probability $r = 0.7$ with $p^{(k+1)} = (1-r)\,P^\top p^{(k)} + r\,p^{(0)}$. Sensitivity to the restart values of the original NetCore ($0.8$) and DADA ($0.3$) papers is reported by `06_compare_degree_bias_method.py`.

## Networks and disease sets

| Network | $G_{cc}$ | $G_{pro}$ |
|---|---|---|
| Recon3D | 2,788 nodes, 22,439 edges | 2,894 nodes (106 pathway nodes), 31,360 edges |
| KEGG (human metabolism) | 3,048 nodes, 18,854 edges | 3,127 nodes (79 pathway nodes), 24,663 edges |

Three disease–metabolite sets are built: **HMDB+CTD**, **MarkerDB**, and **SMPDB** (disease pathways). HMDB and SMPDB entries are restricted to metabolites with HMDB status *detected* and/or *quantified*. Currency metabolites are removed from all three sets, and generic disease terms (e.g. "cancer", "inflammation") from HMDB+CTD and MarkerDB. Diseases with fewer than 8 metabolites mapped to the network are discarded. SMPDB disease names are deduplicated in three steps (normalized name, sorted tokens, token Jaccard ≥ 0.85).

## Evaluation

* Leave-one-out per disease: each known metabolite is held out in turn, and the remaining ones are used as seeds. Seeds are excluded from the candidate list.
* Metrics: AUC, MRR, and Recall@k (k = 5, 10, 20), averaged per disease.
* Statistics: two-sided Wilcoxon signed-rank test paired by disease, with Bonferroni correction over the five metrics within each comparison. Bootstrap 95% confidence intervals use 1,000 resamples (seed 42).

## Data

Download the following files and place them in Google Drive under `MyDrive/CTQW for metabolites/`. Alternatively, set `BASE_DIR` in `src/config.py` to another folder.

| File | Source |
|---|---|
| `Recon3D.json` | [BiGG Models](http://bigg.ucsd.edu/models/Recon3D) |
| `hmdb_metabolites.zip` | [HMDB](https://hmdb.ca/downloads) (All Metabolites, XML) |
| `CTD_chemicals_diseases.csv.gz` | [CTD](https://ctdbase.org/downloads/) |
| `all_chemicals.xml` | [MarkerDB](https://markerdb.ca/downloads) |
| `smpdb_pathways.csv.zip` | [SMPDB](https://smpdb.ca/downloads) |
| `smpdb_metabolites.csv.zip` | [SMPDB](https://smpdb.ca/downloads) |

**KEGG.** KEGG data are not downloaded manually. On the first run, `src/kegg_graph.py` retrieves them from the [KEGG REST API](https://www.kegg.jp/kegg/rest/keggapi.html) and caches them in `results/cache/`. It collects the human metabolic pathways of BRITE `br08901`, excluding global/overview and chemical-structure maps, together with their reactions and reaction equations. Because KEGG is updated regularly, a later retrieval may produce a slightly different network. The results in the manuscript use data retrieved on **YYYY-MM-DD** (86 pathways; 4,316 of 4,496 reactions with a parsable equation).

## Usage

### Google Colab (recommended)

Open one of the notebooks via the badges above and run all cells. Each notebook:

1. clones this repository;
2. mounts Google Drive;
3. runs the experiment scripts for one network.

### Local

```bash
git clone https://github.com/TruongDuyLongPTIT/CTQW-PRO-Quantum-Walk-Based-Metabolite-Disease-Prioritization.git
cd CTQW-PRO-Quantum-Walk-Based-Metabolite-Disease-Prioritization
pip install numpy scipy pandas networkx scikit-learn tqdm
# set BASE_DIR in src/config.py, then for example:
python experiments/01_main_results.py
```

Parsed data and eigendecompositions are cached in `results/cache/`, so only the first run parses HMDB (several minutes). Each script takes from a few minutes to about an hour on a Colab CPU.

## Repository structure

```
├── main_notebook_recon3d.ipynb   # runs all experiments on Recon3D
├── main_notebook_kegg.ipynb      # runs all experiments on KEGG
├── src/
│   ├── config.py                 # paths, hyperparameters, currency metabolite lists
│   ├── utils.py                  # ID and name normalization
│   ├── graph.py                  # Recon3D parsing, G_cc / G_pro construction, eigendecomposition
│   ├── kegg_graph.py             # KEGG retrieval (REST API) and HMDB-KEGG mapping
│   ├── eval_sets.py              # HMDB+CTD, MarkerDB, SMPDB disease sets
│   ├── methods.py                # all ranking methods
│   └── evaluation.py             # leave-one-out, metrics, Wilcoxon, bootstrap, win counts
└── experiments/
    ├── 01_main_results[_kegg].py               # RWR vs CTQW (G_cc); PROFANCY vs CTQW-PRO (G_pro); NH-CTQW-PRO
    ├── 02_ablation_graph[_kegg].py             # effect of removing currency metabolites from G_pro
    ├── 04_biological_interpretability.py       # top-20 predictions for four inborn errors of metabolism (Recon3D, SMPDB)
    ├── 05_supplementary[_kegg].py              # gamma grid search (NH-CTQW-PRO); t grid search (CTQW-PRO, Recon3D)
    └── 06_compare_degree_bias_method[_kegg].py # comparison with degree-bias mitigation methods; degree / coreness of currency metabolites
```

Scripts suffixed `_kegg` run the same experiment on the KEGG network.

## Citation

If you use this code, please cite:

```bibtex
@article{ctqwpro,
  title   = {CTQW-PRO: Quantum Walk-Based Metabolite--Disease Prioritization},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## References

* Köhler S, et al. Walking the interactome for prioritization of candidate disease genes. *Am J Hum Genet.* 2008;82(4):949–958. [doi:10.1016/j.ajhg.2008.02.013](https://doi.org/10.1016/j.ajhg.2008.02.013)
* Shang D, et al. Prioritizing candidate disease metabolites based on global functional relationships between metabolites in the context of metabolic pathways. *PLoS ONE.* 2014;9(8):e104934. [doi:10.1371/journal.pone.0104934](https://doi.org/10.1371/journal.pone.0104934)
* Erten S, et al. DADA: degree-aware algorithms for network-based disease gene prioritization. *BioData Min.* 2011;4:19. [doi:10.1186/1756-0381-4-19](https://doi.org/10.1186/1756-0381-4-19)
* Barel G, Herwig R. NetCore: a network propagation approach using node coreness. *Nucleic Acids Res.* 2020;48(17):e98. [doi:10.1093/nar/gkaa639](https://doi.org/10.1093/nar/gkaa639)
* Frainay C, et al. MetaboRank: network-based recommendation system to interpret and enrich metabolomics results. *Bioinformatics.* 2019;35(2):274–283. [doi:10.1093/bioinformatics/bty577](https://doi.org/10.1093/bioinformatics/bty577)
* Boyd SS, Slawson C, Thompson JA. AMEND 2.0: module identification and multi-omic data integration with multiplex-heterogeneous graphs. *BMC Bioinformatics.* 2025;26:39. [doi:10.1186/s12859-025-06063-x](https://doi.org/10.1186/s12859-025-06063-x)
* Brunk E, et al. Recon3D: a resource enabling a three-dimensional view of gene variation in human metabolism. *Nat Biotechnol.* 2018;36(3):272–281. [doi:10.1038/nbt.4072](https://doi.org/10.1038/nbt.4072)
* Kanehisa M, et al. KEGG: biological systems database as a model of the real world. *Nucleic Acids Res.* 2025;53(D1):D672–D677. [doi:10.1093/nar/gkae909](https://doi.org/10.1093/nar/gkae909)

## License

This project is released under the MIT License (see [LICENSE](LICENSE)).
