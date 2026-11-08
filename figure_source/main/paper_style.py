from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
import subprocess
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'rendered'
OUT = BASE / 'figures'
INK = '#000000'
ORANGE = '#EE7733'
BLUE = '#0077BB'
CYAN = '#33BBEE'
MAGENTA = '#EE3377'
TEAL = '#009988'
RED = '#CC3311'
GRAY = '#BBBBBB'
DARK_GRAY = '#555555'
WHITE = '#FFFFFF'
COLORS = {'selected': ORANGE, 'baseline': BLUE, 'control': GRAY, 'ink': INK}
FONT_LARGE, FONT_NORMAL, FONT_SMALL = (9, 8, 7)
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 8, 'legend.fontsize': 7, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.grid': False, 'axes.unicode_minus': True, 'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'}

def apply_style():
    plt.rcParams.update(STYLE)

def new_figure(width=7.0, height=4.6):
    apply_style()
    return plt.figure(figsize=(width, height))

def clean_axes(ax, zero=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction='out', width=0.6, length=2.5, colors=INK)
    if zero:
        ax.axvline(0, color=GRAY, linewidth=0.5, linestyle='--', zorder=0)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def basis_icon(fig, x, y, metric='J2', size=0.2):
    """0.2-inch two-by-two state icon; x/y are figure coordinates."""
    from matplotlib.patches import Rectangle, Ellipse
    w, h = fig.get_size_inches()
    cw, ch = (size / w / 2, size / h / 2)
    involved = [(0, 0), (0, 1)] if metric in ['J1', 'G_init'] else [(0, 0), (0, 1), (1, 0), (1, 1)]
    for col in range(2):
        for row in range(2):
            rect = Rectangle((x + col * cw, y + row * ch), cw, ch, transform=fig.transFigure, facecolor=WHITE, edgecolor=INK, linewidth=0.6)
            fig.patches.append(rect)
            if (col, row) in involved:
                fig.patches.append(Ellipse((x + (col + 0.5) * cw, y + (row + 0.5) * ch), 1.8 / 72 / w, 1.8 / 72 / h, transform=fig.transFigure, facecolor=INK, edgecolor='none'))
apply_style()
