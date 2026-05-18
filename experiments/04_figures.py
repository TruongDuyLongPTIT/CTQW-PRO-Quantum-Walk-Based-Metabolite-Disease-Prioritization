"""
04_figures.py — Nature-quality figures.

Figures:
    Fig 1: Bar chart — AUC/MRR/R@20 so sánh PROFANCY vs CTQW-PRO vs Driven
    Fig 2: Scatter plot — per-disease MRR (PROFANCY vs CTQW-PRO)
    Fig 3: MRR distribution (violin/box) trên 3 eval sets
    Fig 4: Robustness — original vs clean graph

Chạy: python experiments/04_figures.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy import stats

from config import RESULTS_DIR

# ── Nature-style settings ─────────────────────────────────────────────────────
rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         7,
    'axes.labelsize':    8,
    'axes.titlesize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'figure.dpi':        300,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'lines.linewidth':   1.2,
    'pdf.fonttype':      42,   # editable text in PDF
    'svg.fonttype':      'none',
})

# Nature color palette (from Nature Methods style guide)
COLORS = {
    'PROFANCY':       '#E87722',   # orange
    'CTQW-PRO':       '#2196A6',   # teal
    'Driven CTQW-PRO':'#1D3557',   # dark navy
    'RWR':            '#E87722',
    'CTQW':           '#2196A6',
    'set1':           '#E63946',   # HMDB+CTD
    'set2':           '#457B9D',   # MarkerDB
    'set3':           '#2D6A4F',   # SMPDB
}

# Nature single-column: 88mm = 3.46 in
# Nature double-column: 180mm = 7.09 in
SINGLE  = 3.46
DOUBLE  = 7.09
ONE_HALF= 5.20

FIGS_DIR = RESULTS_DIR / 'figures'
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name, tight=True):
    """Save figure as PDF + PNG."""
    if tight:
        fig.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        out = FIGS_DIR / f'{name}.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f'  Saved: {name}.pdf/.png')
    plt.close(fig)


def _sig_bracket(ax, x1, x2, y, h, p, fontsize=6):
    """Draw significance bracket."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, c='k')
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')
    ax.text((x1+x2)/2, y+h, sig, ha='center', va='bottom',
            fontsize=fontsize, color='k')


# ═════════════════════════════════════════════════════════════════════════════
# Load results
# ═════════════════════════════════════════════════════════════════════════════
results_csv = RESULTS_DIR / 'main_results.csv'
if not results_csv.exists():
    print(f'ERROR: {results_csv} not found. Run 01_main_results.py first.')
    sys.exit(1)

df_all = pd.read_csv(results_csv)
wx_csv = RESULTS_DIR / 'wilcoxon_results.csv'
df_wx  = pd.read_csv(wx_csv) if wx_csv.exists() else None

# Helper
def get_df(table, method, source):
    mask = ((df_all['table'] == table) &
            (df_all['method'] == method) &
            (df_all['source'] == source))
    return df_all[mask]

def get_mean(table, method, source, metric):
    df = get_df(table, method, source)
    return float(df[metric].mean()) if not df.empty else np.nan

def get_p(method_a, method_b, source, metric='mrr'):
    """Get p_bonf from wilcoxon results."""
    if df_wx is None: return np.nan
    mask = ((df_wx['source'] == source) &
            (df_wx['metric'] == metric))
    row = df_wx[mask]
    return float(row['p_bonf'].values[0]) if not row.empty else np.nan


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Grouped bar chart: 3 metrics × 3 sets × 3 methods (Table 2)
# ═════════════════════════════════════════════════════════════════════════════
print('Figure 1: Grouped bar chart')

SETS    = ['HMDB+CTD', 'MarkerDB', 'SMPDB']
METHODS = ['PROFANCY', 't=0.1', 'driven_s2_a0.5']
MLABELS = ['PROFANCY', 'CTQW-PRO', 'Driven CTQW-PRO']
MCOLORS = [COLORS['PROFANCY'], COLORS['CTQW-PRO'], COLORS['Driven CTQW-PRO']]
METRICS = ['auc', 'mrr', 'r@20']
MLABELS_METRIC = ['AUC', 'MRR', 'Recall@20']

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))

for ax, met, met_label in zip(axes, METRICS, MLABELS_METRIC):
    x    = np.arange(len(SETS))
    w    = 0.22
    offs = [-w, 0, w]

    for j, (meth, mlabel, mc) in enumerate(zip(METHODS, MLABELS, MCOLORS)):
        vals = [get_mean('table2', meth, s, met) for s in SETS]
        bars = ax.bar(x + offs[j], vals, w, label=mlabel,
                      color=mc, linewidth=0.5, edgecolor='white',
                      zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(['HMDB\n+CTD', 'MarkerDB', 'SMPDB'], fontsize=7)
    ax.set_ylabel(met_label)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Significance annotation on SMPDB (most informative set)
    if met == 'mrr':
        # CTQW-PRO vs PROFANCY
        v_p  = get_mean('table2', 'PROFANCY', 'SMPDB', met)
        v_c  = get_mean('table2', 't=0.1',   'SMPDB', met)
        v_d  = get_mean('table2', 'driven_s2_a0.5', 'SMPDB', met)
        ymax = max(v_p, v_c, v_d) * 1.05
        ax.set_ylim(0, ymax * 1.25)
        # Star above SMPDB bars
        p_ctqw = get_p('CTQW-PRO', 'PROFANCY', 'SMPDB', 'mrr')
        if not np.isnan(p_ctqw) and p_ctqw < 0.05:
            _sig_bracket(ax, x[2]+offs[0], x[2]+offs[1],
                         ymax, ymax*0.04, p_ctqw)

axes[0].legend(loc='upper left', frameon=False,
               handlelength=1.0, handleheight=0.8)
fig.suptitle('Metabolite–disease prioritization performance',
             fontsize=8, y=1.01)
_save(fig, 'fig1_main_results')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Scatter: per-disease MRR, PROFANCY vs CTQW-PRO (SMPDB)
# ═════════════════════════════════════════════════════════════════════════════
print('Figure 2: Per-disease scatter')

df_p3 = get_df('table2', 'PROFANCY', 'SMPDB')
df_c3 = get_df('table2', 't=0.1',   'SMPDB')

if not df_p3.empty and not df_c3.empty:
    shared = sorted(set(df_p3['disease']) & set(df_c3['disease']))
    mrr_p  = df_p3.set_index('disease').loc[shared, 'mrr'].values
    mrr_c  = df_c3.set_index('disease').loc[shared, 'mrr'].values

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
    ax.scatter(mrr_p, mrr_c, s=12, alpha=0.6, linewidths=0,
               c=COLORS['CTQW-PRO'], zorder=3)

    lim = max(mrr_p.max(), mrr_c.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='y = x')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('PROFANCY MRR')
    ax.set_ylabel('CTQW-PRO MRR')
    ax.set_title(f'SMPDB (n={len(shared)} diseases)')

    # Annotate wins
    n_ctqw = (mrr_c > mrr_p).sum()
    n_prof = (mrr_p > mrr_c).sum()
    ax.text(0.97, 0.03,
            f'CTQW-PRO better: {n_ctqw}/{len(shared)}',
            ha='right', va='bottom', transform=ax.transAxes,
            fontsize=6, color=COLORS['CTQW-PRO'])
    _save(fig, 'fig2_scatter_smpdb')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Violin plot: MRR distribution across 3 sets
# ═════════════════════════════════════════════════════════════════════════════
print('Figure 3: Violin MRR distribution')

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0), sharey=False)

SET_LABELS = ['HMDB+CTD', 'MarkerDB', 'SMPDB']
for ax, src, scol in zip(axes, SET_LABELS,
                          [COLORS['set1'], COLORS['set2'], COLORS['set3']]):
    data_p = get_df('table2', 'PROFANCY', src)['mrr'].values
    data_c = get_df('table2', 't=0.1',   src)['mrr'].values
    data_d = get_df('table2', 'driven_s2_a0.5', src)['mrr'].values

    data   = [d for d in [data_p, data_c, data_d] if len(d) > 0]
    labels = [l for l, d in zip(['PROFANCY','CTQW-PRO','Driven'],
                                  [data_p, data_c, data_d]) if len(d) > 0]
    colors_v = [COLORS['PROFANCY'], COLORS['CTQW-PRO'],
                COLORS['Driven CTQW-PRO']][:len(data)]

    if not data:
        continue

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=False)
    for i, (pc, c) in enumerate(zip(parts['bodies'], colors_v)):
        pc.set_facecolor(c); pc.set_alpha(0.7); pc.set_linewidth(0)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(1.2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
    ax.set_ylabel('MRR' if src == 'HMDB+CTD' else '')
    ax.set_title(src.replace('+', '\n+') if src == 'HMDB+CTD' else src)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5, zorder=0)

fig.suptitle('MRR distribution per disease', fontsize=8, y=1.01)
_save(fig, 'fig3_violin_mrr')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Heatmap: metric comparison table (clean summary)
# ═════════════════════════════════════════════════════════════════════════════
print('Figure 4: Metric heatmap summary')

# Build summary table: rows = method×set, cols = metrics
METS_SHOW   = ['auc', 'mrr', 'r@5', 'r@10', 'r@20']
METS_LABELS = ['AUC', 'MRR', 'R@5', 'R@10', 'R@20']

rows_data = []
rows_label = []
for src in ['HMDB+CTD', 'MarkerDB', 'SMPDB']:
    for meth, mlabel in [('PROFANCY','PROFANCY'),
                          ('t=0.1','CTQW-PRO'),
                          ('driven_s2_a0.5','Driven')]:
        df = get_df('table2', meth, src)
        if df.empty: continue
        row = [float(df[m].mean()) for m in METS_SHOW]
        rows_data.append(row)
        rows_label.append(f'{src}\n{mlabel}')

if rows_data:
    mat  = np.array(rows_data)
    # Normalize per column for color
    mat_n = (mat - mat.min(0)) / (mat.max(0) - mat.min(0) + 1e-9)

    fig, ax = plt.subplots(figsize=(ONE_HALF, len(rows_data)*0.38 + 0.5))
    im = ax.imshow(mat_n, aspect='auto', cmap='Blues', vmin=0, vmax=1)

    for i in range(len(rows_data)):
        for j in range(len(METS_SHOW)):
            ax.text(j, i, f'{mat[i,j]:.3f}',
                    ha='center', va='center', fontsize=6,
                    color='white' if mat_n[i,j] > 0.6 else 'black')

    ax.set_xticks(range(len(METS_SHOW)))
    ax.set_xticklabels(METS_LABELS)
    ax.set_yticks(range(len(rows_label)))
    ax.set_yticklabels(rows_label, fontsize=6)

    # Draw horizontal separators between sets
    for sep in [2.5, 5.5]:
        if sep < len(rows_data):
            ax.axhline(sep, color='white', lw=1.5)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label='Normalized score')
    ax.set_title('Performance summary', fontsize=8)
    _save(fig, 'fig4_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 — R@k curve: precision-recall profile
# ═════════════════════════════════════════════════════════════════════════════
print('Figure 5: R@k curve')

K_VALS = [5, 10, 20, 50]
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0), sharey=False)

for ax, src in zip(axes, ['HMDB+CTD', 'MarkerDB', 'SMPDB']):
    for meth, mlabel, mc in [
        ('PROFANCY',       'PROFANCY',       COLORS['PROFANCY']),
        ('t=0.1',          'CTQW-PRO',       COLORS['CTQW-PRO']),
        ('driven_s2_a0.5', 'Driven CTQW-PRO',COLORS['Driven CTQW-PRO']),
    ]:
        df = get_df('table2', meth, src)
        if df.empty: continue
        rk_vals = [float(df[f'r@{k}'].mean()) for k in K_VALS]
        ax.plot(K_VALS, rk_vals, 'o-', color=mc, markersize=3,
                label=mlabel, linewidth=1.2)

    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k' if src == 'HMDB+CTD' else '')
    ax.set_title(src)
    ax.set_xticks(K_VALS)
    ax.grid(linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

axes[0].legend(loc='upper left', frameon=False, fontsize=6)
fig.suptitle('Recall@k across evaluation sets', fontsize=8, y=1.01)
_save(fig, 'fig5_recall_at_k')


print(f'\nAll figures saved to: {FIGS_DIR}')
print('Done.')
