# ════════════════════════════════════════════════════════════════
# CELL FIGURES v2 — Improved publication figures
# ════════════════════════════════════════════════════════════════
%matplotlib inline
import sys
from pathlib import Path
sys.path.insert(0, '/content/project/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings; warnings.filterwarnings('ignore')
from config import RESULTS_DIR

# ── Style ─────────────────────────────────────────────────────
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'legend.frameon': False,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 1.2, 'patch.linewidth': 0.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

C = {
    'PROFANCY': '#E87C1E', 'CTQW': '#1A7DC4', 'Driven': '#28996B',
    'gray': '#888888', 'light': '#F5F5F5',
    'HMDB': '#E63946', 'MarkerDB': '#457B9D', 'SMPDB': '#2D6A4F',
}
W1 = 3.46; W15 = 5.2; W2 = 7.09
SETS = ['HMDB+CTD', 'MarkerDB', 'SMPDB']

FIGS_DIR = RESULTS_DIR / 'figures'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS_DIR / f'{name}.{ext}', dpi=300)

# ── Data helpers ──────────────────────────────────────────────
df_all = pd.read_csv(RESULTS_DIR / 'main_results.csv')
df_wx  = pd.read_csv(RESULTS_DIR / 'wilcoxon_results.csv') \
         if (RESULTS_DIR / 'wilcoxon_results.csv').exists() else None

TABLE_MAP = {
    'PROFANCY': 'table2', 't=0.1': 'table2',
    'driven_s2_a0.5': 'table3', 'RWR': 'table1', 'CTQW': 'table1',
}

def gdf(method, source):
    tbl = TABLE_MAP.get(method, 'table2')
    return df_all[(df_all['table'] == tbl) & (df_all['method'] == method)
                  & (df_all['source'] == source)]

def gmean(method, source, metric):
    d = gdf(method, source)
    return float(d[metric].mean()) if not d.empty else np.nan

def gse(method, source, metric):
    d = gdf(method, source)
    return float(d[metric].sem()) if len(d) > 1 else 0.0

def get_p(source, metric, row_idx=0):
    if df_wx is None: return np.nan
    r = df_wx[(df_wx['source'] == source) & (df_wx['metric'] == metric)]
    return float(r['p_bonf'].values[row_idx]) if len(r) > row_idx else np.nan

def sig_label(p):
    if np.isnan(p): return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

def sig_bracket(ax, x1, x2, y, h, text, fs=6):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.7, c='#444', clip_on=False)
    ax.text((x1+x2)/2, y+h*1.05, text, ha='center', va='bottom', fontsize=fs, color='#444')

# ════════════════════════════════════════════════════════════════
# FIG 1 — Grouped bar with SEM error bars + value labels
# ════════════════════════════════════════════════════════════════
print('Fig 1: grouped bar + SEM + value labels')

MCFG = [
    ('PROFANCY',        'PROFANCY',       C['PROFANCY'], '///'),
    ('CTQW-PRO',        't=0.1',          C['CTQW'],     ''),
    ('Driven CTQW-PRO', 'driven_s2_a0.5', C['Driven'],   '..'),
]
METS = [('auc', 'AUC'), ('mrr', 'MRR'), ('r@20', 'Recall@20')]

fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw={'wspace': 0.42})

for ax, (met, met_lbl) in zip(axes, METS):
    x = np.arange(len(SETS))
    w = 0.22
    offsets = np.array([-w, 0, w])
    ymax_global = 0

    for j, (lbl, mkey, col, htch) in enumerate(MCFG):
        vals = np.array([gmean(mkey, s, met) for s in SETS])
        errs = np.array([gse(mkey, s, met) for s in SETS])
        vp = np.nan_to_num(vals)
        ep = np.nan_to_num(errs)

        bars = ax.bar(x + offsets[j], vp, w,
                      color=col, hatch=htch,
                      edgecolor='white', linewidth=0.3,
                      label=lbl, zorder=3)
        ax.errorbar(x + offsets[j], vp, yerr=ep,
                    fmt='none', ecolor='#333', elinewidth=0.8,
                    capsize=2.0, capthick=0.8, zorder=4)
        ymax_global = max(ymax_global, (vp + ep).max())

        # Value labels on SMPDB bars only (rightmost, most impressive)
        for xi, (v, e) in enumerate(zip(vp, ep)):
            if xi == 2 and v > 0.01:  # SMPDB only
                ax.text(xi + offsets[j], v + e + ymax_global*0.015,
                        f'{v:.2f}', ha='center', va='bottom',
                        fontsize=5.5, color=col, fontweight='bold')

    # Significance brackets (SMPDB, MRR only to keep clean)
    if met == 'mrr':
        vp_s = gmean('PROFANCY', 'SMPDB', met)
        vc_s = gmean('t=0.1', 'SMPDB', met)
        vd_s = gmean('driven_s2_a0.5', 'SMPDB', met)
        top = max(vp_s, vc_s, vd_s)
        ax.set_ylim(0, top * 1.65)
        p1 = get_p('SMPDB', 'mrr', 0)
        sig_bracket(ax, x[2]+offsets[0], x[2]+offsets[1], top*1.12, top*0.09, sig_label(p1))
        if df_wx is not None:
            r = df_wx[(df_wx['source']=='SMPDB')&(df_wx['metric']=='mrr')]
            if len(r) > 1:
                p2 = float(r['p_bonf'].values[1])
                sig_bracket(ax, x[2]+offsets[1], x[2]+offsets[2], top*1.32, top*0.09, sig_label(p2))
    else:
        ax.set_ylim(0, ymax_global * 1.18)

    ax.set_ylabel(met_lbl, labelpad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(['HMDB\n+CTD', 'MarkerDB', 'SMPDB'], fontsize=6.5)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', linewidth=0.35, alpha=0.45, zorder=0, linestyle=':')
    ax.set_axisbelow(True)

axes[0].legend(loc='upper left', fontsize=6, handlelength=1.0,
               handletextpad=0.4, borderpad=0.2, labelspacing=0.3,
               frameon=True, framealpha=0.85, edgecolor='#ddd', linewidth=0.4)
fig.suptitle('Metabolite–disease prioritization performance', fontsize=8, y=1.04)
plt.show(); savefig(fig, 'fig1_bar')

# ════════════════════════════════════════════════════════════════
# FIG 2 — Scatter + marginal KDE + color by Δ
# ════════════════════════════════════════════════════════════════
print('\nFig 2: scatter + marginal KDE')

dfp = gdf('PROFANCY', 'SMPDB').set_index('disease')
dfc = gdf('t=0.1',   'SMPDB').set_index('disease')
shared = sorted(set(dfp.index) & set(dfc.index))
mp  = dfp.loc[shared, 'mrr'].values
mc  = dfc.loc[shared, 'mrr'].values
delta = mc - mp  # positive → CTQW wins

fig = plt.figure(figsize=(W1 + 0.5, W1 + 0.5))
ax_main = fig.add_axes([0.15, 0.15, 0.62, 0.62])
ax_top  = fig.add_axes([0.15, 0.78, 0.62, 0.15])
ax_right= fig.add_axes([0.78, 0.15, 0.15, 0.62])

# Color by delta
norm = plt.Normalize(delta.min(), delta.max())
cmap = plt.cm.RdYlGn
sc = ax_main.scatter(mp, mc, c=delta, cmap=cmap, norm=norm,
                     s=14, alpha=0.75, linewidths=0.3, edgecolors='white', zorder=3)

lim = max(mp.max(), mc.max()) * 1.1
ax_main.plot([0, lim], [0, lim], '--', lw=0.9, c='#aaa', zorder=2)
ax_main.fill_between([0, lim], [0, lim], lim, alpha=0.04, color=C['CTQW'])
ax_main.fill_between([0, lim], 0, [0, lim], alpha=0.04, color=C['PROFANCY'])
ax_main.set_xlim(-0.005, lim); ax_main.set_ylim(-0.005, lim)
ax_main.set_xlabel('PROFANCY  MRR', labelpad=3)
ax_main.set_ylabel('CTQW-PRO  MRR', labelpad=3)
n_win = (delta > 0).sum()
ax_main.text(0.97, 0.04,
             f'CTQW-PRO better:\n{n_win}/{len(shared)} ({100*n_win/len(shared):.0f}%)',
             ha='right', va='bottom', transform=ax_main.transAxes,
             fontsize=6, color='#2D6A4F', linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', lw=0.5))

# Marginal KDE
from scipy.stats import gaussian_kde
for vals, ax_m, vertical in [(mp, ax_top, False), (mc, ax_right, True)]:
    kde = gaussian_kde(vals, bw_method=0.3)
    xr = np.linspace(0, lim, 200)
    yr = kde(xr)
    if vertical:
        ax_m.fill_betweenx(xr, 0, yr, alpha=0.3, color=C['CTQW'] if ax_m==ax_right else C['PROFANCY'])
        ax_m.plot(yr, xr, lw=0.8, color=C['CTQW'])
        ax_m.set_xlim(left=0); ax_m.set_ylim(-0.005, lim)
    else:
        ax_m.fill_between(xr, 0, yr, alpha=0.3, color=C['PROFANCY'])
        ax_m.plot(xr, yr, lw=0.8, color=C['PROFANCY'])
        ax_m.set_ylim(bottom=0); ax_m.set_xlim(-0.005, lim)
    for spine in ax_m.spines.values(): spine.set_visible(False)
    ax_m.set_xticks([]); ax_m.set_yticks([])

# Colorbar
cax = fig.add_axes([0.79, 0.82, 0.12, 0.03])
cb  = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label('Δ MRR', fontsize=5.5, labelpad=2)
cb.ax.tick_params(labelsize=5)

for spine in ['top', 'right']:
    ax_main.spines[spine].set_visible(False)
plt.show(); savefig(fig, 'fig2_scatter')

# ════════════════════════════════════════════════════════════════
# FIG 3 — Violin + strip plot (individual points)
# ════════════════════════════════════════════════════════════════
print('\nFig 3: violin + strip')

MKEYS = ['PROFANCY', 't=0.1', 'driven_s2_a0.5']
MLBLS = ['PROFANCY', 'CTQW-PRO', 'Driven']
MCOLS = [C['PROFANCY'], C['CTQW'], C['Driven']]

fig, axes = plt.subplots(1, 3, figsize=(W2, 2.1), gridspec_kw={'wspace': 0.38})
rng = np.random.default_rng(42)

for ax, src in zip(axes, SETS):
    data_list = [gdf(mk, src)['mrr'].values for mk in MKEYS]
    data_ok   = [(d, l, c) for d, l, c in zip(data_list, MLBLS, MCOLS) if len(d) > 1]
    if not data_ok: continue

    # Violin
    parts = ax.violinplot([d for d, _, _ in data_ok],
                          positions=range(len(data_ok)),
                          showmedians=False, showextrema=False, widths=0.65)
    for body, (_, _, c) in zip(parts['bodies'], data_ok):
        body.set_facecolor(c); body.set_alpha(0.30)
        body.set_edgecolor(c); body.set_linewidth(0.8)

    for i, (d, _, c) in enumerate(data_ok):
        # IQR box
        q25, med, q75 = np.percentile(d, [25, 50, 75])
        ax.fill_between([i-0.12, i+0.12], [q25, q25], [q75, q75],
                        color=c, alpha=0.45, linewidth=0, zorder=3)
        # Median line
        ax.plot([i-0.14, i+0.14], [med, med], lw=2.0, c='white', solid_capstyle='round', zorder=4)
        ax.plot([i-0.14, i+0.14], [med, med], lw=1.0, c=c, solid_capstyle='round', zorder=5)
        # Mean marker
        ax.scatter([i], [d.mean()], marker='D', s=12, color='white',
                   edgecolors=c, linewidths=0.8, zorder=6)
        # Jittered strip
        jitter = rng.uniform(-0.08, 0.08, len(d))
        ax.scatter(i + jitter, d, s=4, color=c, alpha=0.45,
                   linewidths=0, zorder=2)

    ax.set_xticks(range(len(data_ok)))
    ax.set_xticklabels([l for _, l, _ in data_ok], rotation=28, ha='right', fontsize=6)
    ax.set_ylabel('MRR' if src == 'HMDB+CTD' else '', labelpad=3)
    ax.set_title(src, fontsize=7.5, fontweight='bold')
    ax.grid(axis='y', linewidth=0.35, alpha=0.45, zorder=0, linestyle=':')
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=-0.01)

# Legend
legend_handles = [mpatches.Patch(color=c, label=l, alpha=0.7)
                  for _, l, c in zip(MLBLS, MLBLS, MCOLS)]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('MRR distribution per disease', fontsize=8, y=1.02)
plt.show(); savefig(fig, 'fig3_violin')

# ════════════════════════════════════════════════════════════════
# FIG 4 — Recall@k curves + SE shading
# ════════════════════════════════════════════════════════════════
print('\nFig 4: Recall@k + SE bands')

K_VALS = [5, 10, 20, 50]
MSTYLES = [
    ('PROFANCY',       'PROFANCY',        C['PROFANCY'], '--', 's'),
    ('t=0.1',          'CTQW-PRO',        C['CTQW'],     '-',  'o'),
    ('driven_s2_a0.5', 'Driven CTQW-PRO', C['Driven'],   '-',  '^'),
]

fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw={'wspace': 0.40})

for ax, src in zip(axes, SETS):
    for mkey, mlbl, col, ls, mk in MSTYLES:
        d = gdf(mkey, src)
        if d.empty: continue
        means = np.array([float(d[f'r@{k}'].mean()) for k in K_VALS])
        sems  = np.array([float(d[f'r@{k}'].sem())  for k in K_VALS])
        ax.plot(K_VALS, means, ls, color=col, markersize=4,
                marker=mk, label=mlbl, linewidth=1.3,
                markerfacecolor='white', markeredgewidth=1.0, zorder=4)
        ax.fill_between(K_VALS, means - sems, means + sems,
                        color=col, alpha=0.12, linewidth=0, zorder=2)
    ax.set_xlabel('k', labelpad=2)
    ax.set_ylabel('Recall@k' if src == 'HMDB+CTD' else '', labelpad=3)
    ax.set_title(src, fontsize=7.5, fontweight='bold')
    ax.set_xticks(K_VALS); ax.set_xticklabels(K_VALS, fontsize=6.5)
    ax.grid(linewidth=0.35, alpha=0.45, zorder=0, linestyle=':')
    ax.set_axisbelow(True); ax.set_ylim(bottom=0)

axes[0].legend(loc='upper left', fontsize=6, handlelength=1.5,
               handletextpad=0.4, borderpad=0.2, labelspacing=0.3,
               frameon=True, framealpha=0.85, edgecolor='#ddd', linewidth=0.4)
fig.suptitle('Recall@k across evaluation sets', fontsize=8, y=1.03)
plt.show(); savefig(fig, 'fig4_recall_at_k')

# ════════════════════════════════════════════════════════════════
# FIG 5 — Heatmap with best-cell highlighting + Δ column
# ════════════════════════════════════════════════════════════════
print('\nFig 5: heatmap + best-cell highlight + delta')

METS_H   = ['auc', 'mrr', 'r@5', 'r@10', 'r@20']
METS_LBL = ['AUC', 'MRR', 'R@5', 'R@10', 'R@20']
SET_COLS  = [C['HMDB'], C['MarkerDB'], C['SMPDB']]
MCONFIGS  = [('PROFANCY', 'PROFANCY'), ('t=0.1', 'CTQW-PRO'), ('driven_s2_a0.5', 'Driven')]

rows_d, rows_l, row_colors = [], [], []
for si, src in enumerate(SETS):
    for mkey, mlbl in MCONFIGS:
        d = gdf(mkey, src)
        if d.empty: continue
        rows_d.append([float(d[m].mean()) for m in METS_H])
        rows_l.append(mlbl)
        row_colors.append(SET_COLS[si])

mat   = np.array(rows_d)
# Normalize per column
mat_n = (mat - mat.min(0)) / (mat.max(0) - mat.min(0) + 1e-9)
nrows, ncols = mat.shape

# Compute ΔAUC column: CTQW-PRO minus PROFANCY per set
delta_col = []
for si in range(len(SETS)):
    base = si * 3
    if base+1 < len(rows_d):
        delta_col.append([rows_d[base+1][j] - rows_d[base][j] for j in range(ncols)])
    else:
        delta_col.append([0]*ncols)

fig, ax = plt.subplots(figsize=(W15 + 0.5, nrows*0.40 + 1.0))
im = ax.imshow(mat_n, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest')

# Best cell per column → bold border
best_per_col = mat.argmax(axis=0)
for col_i, row_i in enumerate(best_per_col):
    rect = plt.Rectangle((col_i-0.5, row_i-0.5), 1, 1,
                          fill=False, edgecolor='#FFD700', linewidth=2.0, zorder=5)
    ax.add_patch(rect)

# Cell text
for i in range(nrows):
    for j in range(ncols):
        txt_c = 'white' if mat_n[i, j] > 0.60 else '#222'
        bold  = 'bold' if best_per_col[j] == i else 'normal'
        ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                fontsize=6.5, color=txt_c, fontweight=bold)

# Separator lines between sets
for sep in [2.5, 5.5]:
    ax.axhline(sep, color='white', lw=2.0)

ax.set_xticks(range(ncols)); ax.set_xticklabels(METS_LBL, fontsize=8, fontweight='bold')
ax.set_yticks(range(nrows)); ax.set_yticklabels(rows_l, fontsize=7)

# Left-side colored group labels
for i, c in enumerate(row_colors):
    ax.add_patch(plt.Rectangle((-0.5, i-0.5), 0.08, 1,
                               color=c, transform=ax.get_yaxis_transform(),
                               clip_on=False, zorder=6))

# Right-side set labels
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([1, 4, 7])
ax2.set_yticklabels(['HMDB+CTD', 'MarkerDB', 'SMPDB'], fontsize=7, fontweight='bold')
for tick, col in zip(ax2.get_yticklabels(), SET_COLS):
    tick.set_color(col)
for spine in ax2.spines.values(): spine.set_visible(False)
ax2.tick_params(right=False)

cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.16, shrink=0.8, aspect=18)
cb.ax.tick_params(labelsize=5.5)
cb.set_label('Normalized score', fontsize=6, labelpad=3)
ax.set_title('Performance summary   (★ = best per metric)', fontsize=8, pad=8)

# Legend for gold box
gold_patch = mpatches.Patch(facecolor='none', edgecolor='#FFD700', linewidth=2, label='Best')
ax.legend(handles=[gold_patch], loc='lower right', fontsize=6,
          frameon=True, framealpha=0.9, edgecolor='#ddd', bbox_to_anchor=(1.0, -0.08))
fig.tight_layout(pad=0.6)
plt.show(); savefig(fig, 'fig5_heatmap')

print(f'\n✓ Saved to: {FIGS_DIR}')
