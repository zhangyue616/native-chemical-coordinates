"""Portable rendering of the table-of-contents graphic from frozen layout values and saved query geometry."""
import argparse
import json
import math
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
W, H = (3.25, 1.75)
INK, ORANGE, BLUE, GRAY, WHITE, DIVIDER = ('#000000', '#D55E00', '#0072B2', '#7F7F7F', '#FFFFFF', '#BDBDBD')
HALO_RADIUS_PT, HALO_ALPHA, BOND_GAP_PT = (4.2, 0.25, 0.6)
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 8, 'legend.fontsize': 7, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.grid': False, 'axes.unicode_minus': True, 'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def trim_bond(vertices):
    (x0, y0), (x1, y1) = vertices
    length = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (BOND_GAP_PT * (x1 - x0) / length, BOND_GAP_PT * (y1 - y0) / length)
    return [[x0 + dx, y0 + dy], [x1 - dx, y1 - dy]]


def paint_query(fig, left, bottom, geometry, match_mark):
    width, height = (geometry['width_pt'], geometry['height_pt'])
    ax = fig.add_axes([left / W, bottom / H, width / 72 / W, height / 72 / H])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    if match_mark == 'halo':
        for circle in geometry['circles']:
            ax.add_patch(Circle((circle['x'], circle['y']), HALO_RADIUS_PT, facecolor=ORANGE, edgecolor='none', alpha=HALO_ALPHA, zorder=0.2))
    for path in geometry['paths']:
        vertices = path['vertices']
        if match_mark == 'halo' and 'atom-0 atom-1' in path.get('source_class', ''):
            vertices = trim_bond(vertices)
        patch = PathPatch(MplPath(vertices, path['codes']), facecolor='none', edgecolor=path.get('color', 'black'), linewidth=path.get('linewidth_pt', 0.6), capstyle='round', joinstyle='round')
        if path.get('dash'):
            patch.set_linestyle((0, [float(v) * geometry['source_SVG_to_point_unit'] for v in re.findall(r'\d*\.?\d+', path['dash'])]))
        ax.add_patch(patch)
    for label in geometry['texts']:
        ax.text(label['x'], label['y'], label['text'], fontsize=7, family='Arial', color=label.get('color', 'black'), weight=label.get('font_weight', 'normal'), va='baseline', ha={'start': 'left', 'middle': 'center', 'end': 'right'}.get(label['anchor'], 'left'))
    if match_mark == 'ring':
        for circle in geometry['circles']:
            ax.add_patch(Circle((circle['x'], circle['y']), circle['radius_pt'], fill=False, edgecolor=circle['color'], linewidth=circle['linewidth_pt']))


def render(root, version, match_mark):
    data = read(root / 'source_data/toc_graphic_plot_data.json')
    labels, arrows, glyphs = (data['label_records'], data['arrows'], data['coordinate_glyphs'])
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    def text(record):
        x, y = record['anchor_in']
        ax.text(x, y, record['label'], ha='center', va='center', fontsize=record['size_pt'], color=INK, fontweight='normal', linespacing=1.12)

    def line(start, end, color=INK):
        ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=0.5, color=color)

    def box(x, y, w, h, edge=INK, fill=WHITE, lw=0.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw))

    def arrow(record):
        ax.add_patch(FancyArrowPatch(tuple(record['start_in']), tuple(record['end_in']), arrowstyle='-|>', mutation_scale=5, linewidth=0.6, color=INK, shrinkA=0, shrinkB=0))

    def strip(record, n=5, cell=0.07):
        x, y, _, height = record['bounds_in']
        for j in range(n):
            box(x + j * cell, y, cell, height, edge=GRAY, fill=record['fill'] if j == record['selected_index'] else WHITE, lw=0.5)

    line((1.625, 0.1), (1.625, 1.65), DIVIDER)
    line((0.1, 0.875), (1.51, 0.875), DIVIDER)
    line((1.74, 0.875), (3.15, 0.875), DIVIDER)
    for record in labels[:4]:
        text(record)
    for index, query in enumerate(data['chemical_queries']):
        geometry = read(root / 'figure_source/assets/toc' / ('query_%d_geometry.json' % query['concept_id']))
        paint_query(fig, query['left_in'], query['bottom_in'], geometry, match_mark)
        strip(glyphs[index])
        center = glyphs[index]['bounds_in'][0] + 0.175
        line((center, 1.013), (center, 1.008))
    for record in glyphs[2:8]:
        strip(record)
    arrow(arrows[0])
    for offset, y in enumerate([0.36, 0.14]):
        box(0.24, y, 0.43, 0.16, edge=BLUE)
        text(labels[4 + offset])
        arrow(arrows[1 + offset])
        strip(glyphs[8 + offset])
    for i, x in enumerate([1.84, 2.24, 2.64]):
        box(x, 0.19, 0.29, 0.27)
        if i == 1:
            box(x + 0.1175, 0.285, 0.055, 0.075, edge=GRAY, fill=ORANGE, lw=0.5)
    for record in arrows[3:6]:
        arrow(record)
    ax.add_patch(Circle((3.105, 0.325), 0.021, facecolor=INK, edgecolor=INK, linewidth=0.5))
    strip(glyphs[10])
    arrow(arrows[6])
    target = root / 'rendered/toc_graphic' / version
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'toc_graphic.png').exists():
        raise FileExistsError('Choose a new output version')
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(str(target / ('toc_graphic.' + ext)), dpi=600, facecolor=WHITE, edgecolor=WHITE)
    plt.close(fig)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='local')
    parser.add_argument('--match-mark', choices=['halo', 'ring'], default='halo', help='ring reproduces the earlier open-circle artwork')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(render(args.root, args.version, args.match_mark))
