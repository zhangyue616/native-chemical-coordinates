"""Portable rendering from frozen values and geometry."""
import argparse
import copy
import csv
import colorsys
import json
import math
import re
import shutil
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath, TextToPath
ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'figure_source/assets/figure_1'
import paper_style as ps
from chemical_asset_style import _align_atom_text
BASE = ROOT
OUTPUT_BASE = ROOT / 'rendered'
ASSETS = FROZEN
SOURCE = BASE / 'MANUSCRIPT.md'
LABEL_SOURCE = 'Frozen figure source'
FROZEN_SOURCE = 'Frozen figure source'
TOKEN_EVIDENCE = FROZEN / 'token_strip_evidence.json'
W, BASE_H = (7.0, 4.2)
BOTTOM_EXTENSION = 0.26
H = BASE_H + BOTTOM_EXTENSION
LEGEND_BOTTOM, LEGEND_HEIGHT, LEGEND_GAP = (0.05, 0.16, 0.1)
LEGEND_DELTA = LEGEND_BOTTOM - BOTTOM_EXTENSION - 0.105
INK, MUTED = ('#20252A', '#7A8288')
ORANGE, BLUE, TEAL = ('#EE7733', '#0077BB', '#009988')
MASK = '#0077BB'
J1, J2 = ('#33BBEE', '#EE3377')
CPK = {'N': BLUE, 'O': '#CC3311', 'S': '#CCBB44', 'Cl': TEAL, 'C': INK, 'H': INK}
PANEL_COLORS = ['#FBF8EF', '#F0F7FC', '#F7F0F8', '#F1F7F2']
CHECKPOINTS = ['ChemBERTa-77M-MLM', 'ChemBERTa-77M-MTR', 'CheMeleon', 'MolCLR GIN', 'Hu masking', 'Hu edgepred', 'Hu contextpred', 'Hu infomax', 'MoLFormer']
METHODS = ['Cross-checkpoint audit · Fig. 2', 'Fixed-coordinate transfer · Fig. 3', 'Lookup with and without unit · Fig. 4', 'Chemical activation replacement · Fig. 5A–C']
QUESTIONS = ['Does it beat simple controls?', 'Does it carry to new molecules?', 'Does the unit improve prediction?', 'Can chemistry replace activation?']
DISPLAY_SOURCE = 'Frozen figure source'
DISPLAY_METHODS = ['Fig. 2A–B', 'Fig. 2C–D', 'Fig. 3', 'Fig. 4']
DISPLAY_TEXT = {'SMILES\ntokens': 'SMILES', 'Carbonyl\nquery': 'Carbonyl', 'Held-out\namide': 'Amide', 'Held-out\nketone 1': 'Ketone 1', 'Held-out\nketone 2': 'Ketone 2', 'ESM-2 8M, block 3': 'ESM-2 8M,\nblock 3', 'masked residue': 'mask', 'X / S / L': 'X/S/L', 'X / S / L +': 'X/S/L +', 'Predicted\nactivations': 'Predicted\nactivation', 'token ×\ncoordinate': 'coordinates', 'Frozen\ndownstream\noutputs': 'Frozen\noutputs', 'Original\noutputs': 'original', 'Same-molecule\ndisplacement shuffle': 'Shuffled\ncontrol'}

def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def visible_text_runs(records):
    """Cl and NH are single atom labels even when RDKit emits two SVG glyphs."""
    merged, atoms = ([], {})
    for i, record in enumerate(records):
        if record['role'] == 'chemical_atom_glyph':
            key = (record['source'], record.get('chemical_atom_index'))
            if key in atoms:
                merged[atoms[key]]['text'] += record['text']
                merged[atoms[key]]['raw_text_record_indices'].append(i)
                continue
            atoms[key] = len(merged)
        rec = dict(record)
        rec['raw_text_record_indices'] = [i]
        merged.append(rec)
    return merged

def text_bbox(label):
    prop = FontProperties(family='Arial', size=7)
    box = TextPath((0, 0), label['text'], size=7, prop=prop).get_extents()
    advance = TextToPath().get_text_width_height_descent(label['text'], prop, False)[0]
    x = label['x'] - {'middle': advance / 2, 'end': advance}.get(label.get('anchor'), 0)
    return [x + box.x0, label['y'] - box.y1, x + box.x1, label['y'] - box.y0]

def finalize_geometry(g):
    g = copy.deepcopy(g)
    _align_atom_text(g)
    boxes = [text_bbox(t) for t in g['texts']]
    coords = [xy for p in g['paths'] for xy in p['vertices']]
    boxes += [[x - 0.3, y - 0.3, x + 0.3, y + 0.3] for x, y in coords]
    boxes += [[a['x'] - 3.2, a['y'] - 3.2, a['x'] + 3.2, a['y'] + 3.2] for a in g['atoms'] if a['index'] in g['matched_atom_indices']]
    x0, y0 = (min((b[0] for b in boxes)) - 0.5, min((b[1] for b in boxes)) - 0.5)
    x1, y1 = (max((b[2] for b in boxes)) + 0.5, max((b[3] for b in boxes)) + 0.5)
    for p in g['paths']:
        p['vertices'] = [[x - x0, y - y0] for x, y in p['vertices']]
    for key in ['texts', 'atoms']:
        for item in g[key]:
            item['x'] -= x0
            item['y'] -= y0
    g.update({'width_pt': float(x1 - x0), 'height_pt': float(y1 - y0), 'font_size_pt': 7.0, 'geometry_finalized': True})
    return g

def choose_geometry(stem, available_pt):
    candidates = json.loads((ASSETS / (stem + '_layout_candidates.json')).read_text(encoding='utf-8'))
    options = [finalize_geometry(g) for g in candidates]
    score = lambda g: max(g['width_pt'] / available_pt[0], g['height_pt'] / available_pt[1])
    g = min(options, key=score)
    g['layout_fit'] = {'available_pt': available_pt, 'required_pt': [g['width_pt'], g['height_pt']], 'fits': score(g) <= 1.0, 'fit_ratio': score(g), 'minimum_uniform_point_geometry_bond_pt_tested': 8.0, 'font_reduction_applied': False}
    return g

def build(version, preview=True):
    ps.apply_style()
    plt.rcParams['text.color'] = INK
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(-BOTTOM_EXTENSION, BASE_H)
    ax.axis('off')
    texts, edges, structures, glyphs = ([], [], [], [])
    panel_bounds = [[0.05, 2.175, 3.375, 1.975], [3.575, 2.175, 3.375, 1.975], [0.05, 0.05, 3.375, 1.975], [3.575, 0.05, 3.375, 1.975]]
    token_evidence = json.loads(TOKEN_EVIDENCE.read_text(encoding='utf-8'))
    if token_evidence['tokens'] != ['*', 'C', '(', '=', 'O', ')', '*']:
        raise ValueError('Frozen offline-token evidence differs from frozen seven-cell strip')

    def text(x, y, label, size=7, ha='left', va='center', weight='normal', source='MANUSCRIPT.md:19', role='label', panel=None, color=INK, **kw):
        original_label = label
        if preview:
            label = DISPLAY_TEXT.get(label, label)
            if role == 'method':
                label = DISPLAY_METHODS[panel - 1]
            if role == 'legend' and original_label == 'masked residue':
                label = original_label
            if label != original_label:
                source += '; display wording ' + DISPLAY_SOURCE + ':44-125'
        if role in ['method', 'legend'] or label in ['Predicted\nactivation', 'X', 'S']:
            source += 'Frozen figure source' + FROZEN_SOURCE
        if label in ['Block 1', 'Block 2', 'Block 3', 'ESM-2 8M,\nblock 3', 'L +', 'X/S/L +', 'Shuffled\ncontrol']:
            source += '; ' + LABEL_SOURCE
        item = ax.text(x, y, label, fontsize=size, family='Arial', ha=ha, va=va, fontweight=weight, color=color, linespacing=1.04, **kw)
        texts.append({'artist': item, 'text': label, 'size_pt': size, 'source': source, 'role': role, 'panel': panel, 'anchor_in': [x, y + BOTTOM_EXTENSION], 'original_display_label': original_label})
        return item

    def box(x, y, w, h, fill='white', edge='#A7ADB1', lw=0.65, radius=0.035, **kw):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=' + str(radius), linewidth=lw, edgecolor=edge, facecolor=fill, **kw)
        ax.add_patch(p)
        return p

    def line(points, lw=0.65, color=INK, style='-'):
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=lw, linestyle=style, solid_capstyle='round', zorder=3)

    def arrow(start, end, role, lw=0.7, color=INK, style='-', connectionstyle='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=6, linewidth=lw, color=color, linestyle=style, shrinkA=0, shrinkB=0, connectionstyle=connectionstyle, zorder=4))
        edges.append({'role': role, 'start_in': [start[0], start[1] + BOTTOM_EXTENSION], 'end_in': [end[0], end[1] + BOTTOM_EXTENSION], 'path_endpoint_length_in': math.hypot(end[0] - start[0], end[1] - start[1]), 'linewidth_pt': lw, 'color': color, 'linestyle': style})

    def strip(x, y, w, h, n=7, selected=None, mask=False, role='strip'):
        for j in range(n):
            selected_cell = j == selected
            fill = (MASK if mask else ORANGE) if selected_cell else 'white'
            ax.add_patch(Rectangle((x + j * w / n, y), w / n, h, facecolor=fill, edgecolor='#858E94', linewidth=0.55))
        glyphs.append({'role': role, 'bounds_in': [x, y, w, h], 'n_cells': n, 'selected_index': selected, 'selected_fill': MASK if mask else ORANGE})
    for i, bounds in enumerate(panel_bounds):
        box(*bounds, fill=PANEL_COLORS[i], edge='none', lw=0, radius=0.06)
        x, y, width, height = bounds
        text(x + 0.11, y + height - 0.15, chr(65 + i) + '  ' + QUESTIONS[i], size=8.5, weight='bold', panel=i + 1, role='lettered_question', source=LABEL_SOURCE + '; MANUSCRIPT.md:19')
        text(x + 0.11, y + height - 0.3, METHODS[i], size=7.2, panel=i + 1, role='method', source='MANUSCRIPT.md:19')
    for i, name in enumerate(CHECKPOINTS):
        y = 3.535 - i * 0.148
        if i == 1:
            box(0.16, y - 0.069, 1.69, 0.139, fill='#FBE6D9', edge=ORANGE, lw=0.8, radius=0.05)
        text(0.2, y, name, panel=1, role='checkpoint', source='MANUSCRIPT.md:35-45')
    text(2.43, 3.51, 'Native', ha='center', panel=1)
    text(3.05, 3.51, 'Rotated', ha='center', panel=1)
    text(1.96, 3.18, 'Trained', rotation=90, ha='center', panel=1)
    text(1.96, 2.52, 'Initialization', rotation=90, ha='center', panel=1)
    ax.add_patch(Rectangle((2.12, 2.24), 1.24, 1.16, fill=False, edgecolor='#DDDDDD', linewidth=0.5, zorder=1.1))
    for pts in [[(2.74, 2.24), (2.74, 3.4)], [(2.12, 2.85), (3.36, 2.85)]]:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color='#DDDDDD', linewidth=0.5, zorder=1.1)
    glyphs.append({'role': 'calibration_matrix_grid', 'bounds_in': [2.12, 2.24, 1.24, 1.16], 'rows': 2, 'columns': 2, 'linewidth_pt': 0.5, 'color': '#DDDDDD', 'vertical_divider_x_in': 2.74, 'horizontal_divider_y_in': 2.85})
    for x, y, marker, fill in [(2.43, 3.18, 'o', INK), (3.05, 3.18, 'D', INK), (2.43, 2.52, 'o', 'white'), (3.05, 2.52, 'D', 'white')]:
        ax.plot(x, y, marker=marker, markersize=7.5, markeredgewidth=0.8, markeredgecolor=INK, markerfacecolor=fill, linestyle='None', zorder=5)
    arrow((2.43, 2.63), (2.43, 3.07), 'J1_initialization_native_to_trained_native', lw=1.0, color=J1)
    text(2.28, 2.85, 'J1', ha='center', panel=1, color=J1, bbox={'facecolor': PANEL_COLORS[0], 'edgecolor': 'none', 'pad': 0.4})
    arrow((2.94, 3.18), (2.55, 3.18), 'J2_trained_rotated_to_trained_native', lw=1.0, color=J2)
    text(2.74, 3.33, 'J2', ha='center', panel=1, color=J2, bbox={'facecolor': PANEL_COLORS[0], 'edgecolor': 'none', 'pad': 0.4})
    arrow((2.94, 2.52), (2.55, 2.52), 'initialization_rotated_to_native_reference', lw=0.7, color=MUTED)
    text(2.74, 2.34, 'reference', ha='center', panel=1, color=MUTED, source='MANUSCRIPT.md:56', bbox={'facecolor': PANEL_COLORS[0], 'edgecolor': 'none', 'pad': 0.4})
    text(3.76 if preview else 3.79, 3.6, 'SMILES\ntokens', ha='center', panel=2)
    for n, x in [(1, 4.04), (2, 4.56), (3, 6.4)]:
        box(x, 3.445, 0.44, 0.3, fill='#FFFFFF', edge='#8C969C')
        text(x + 0.22, 3.595, 'Block ' + str(n), ha='center', panel=2)
    arrow((3.97, 3.595), (4.035, 3.595), 'SMILES_to_Block1')
    arrow((4.49, 3.595), (4.55, 3.595), 'Block1_to_Block2')
    arrow((5.005, 3.72), (5.38, 3.66), 'Block2_to_post_GELU')
    strip(5.39, 3.57, 0.81, 0.09, selected=1, role='post_GELU_selected_coordinate')
    text(5.31, 3.595, 'unit', ha='right', panel=2, source=LABEL_SOURCE)
    text(5.795, 3.76, 'FFN post-GELU', ha='center', panel=2)
    arrow((6.21, 3.615), (6.39, 3.595), 'post_GELU_to_Block3')
    strip(5.39, 3.36, 0.81, 0.11, selected=1, role='token_aligned_to_selected_coordinate')
    text(5.31, 3.415, 'tokens', ha='right', panel=2, source=LABEL_SOURCE)
    for j, token in enumerate(token_evidence['tokens']):
        text(5.39 + (j + 0.5) * 0.81 / 7, 3.415, token, ha='center', panel=2, role='smiles_token', source=str(TOKEN_EVIDENCE))
    selected_x = 5.39 + 1.5 * 0.81 / 7
    line([(selected_x, 3.555), (selected_x, 3.475)], lw=0.55, color=ORANGE)
    holders = [(3.685, 0.64, 'Carbonyl\nquery', None), (4.37, 0.82, 'Held-out\namide', 'T1_CID60628745'), (5.235, 0.77, 'Held-out\nketone 1', 'T2_CID43411971'), (6.05, 0.79, 'Held-out\nketone 2', 'T3_CID678545')]
    for x, width, label, stem in holders:
        holder = [x, 2.56 if preview else 2.48, width, 0.77 if preview else 0.85]
        box(*holder, fill='#FFFFFF', edge='#A4ACB1', radius=0.04)
        text(x + width / 2, 3.235 if preview else 3.21, label, ha='center', panel=2)
        if stem is None:
            g = json.loads((FROZEN / 'query_24_geometry.json').read_text(encoding='utf-8'))
            g = finalize_geometry(g)
            text(x + width / 2, 2.66 if preview else 2.62, '[CX3]=[OX1]', ha='center', panel=2, source='sources/chemical_assets/molecules/T1_CID60628745_metadata.json:51')
            plot_chemical(g, x + (width - g['width_pt'] / 72) / 2, 2.8, fig, texts, 2, source='sources/chemical_assets/queries/query_24_geometry.json')
        else:
            g = choose_geometry(stem, [width * 72 - 1, 0.59 * 72])
            plot_chemical(g, x + (width - g['width_pt'] / 72) / 2, 2.58 if preview else 2.51, fig, texts, 2, source='sources/chemical_assets/molecules/' + stem + '_metadata.json:17-18')
        structures.append({'label': label, 'bounds_in': holder, 'asset': stem, 'CID': g.get('metadata', {}).get('CID'), 'SMILES': g.get('metadata', {}).get('SMILES'), 'width_pt': g['width_pt'], 'height_pt': g['height_pt'], 'matched_atom_indices': g['matched_atom_indices'], 'font_pt': 7, 'fit': g.get('layout_fit'), 'source_metadata_file': g.get('source_metadata_file'), 'graph_atom_count': len(g['atoms']), 'rendered_bond_path_count': len(g['paths']), 'carbonyl_halo_count': len(g['matched_atom_indices'])})
    text(3.73, 2.365, 'Fit 5,000', panel=2)
    arrow((4.28, 2.365), (4.53, 2.365), 'fit_locked_to_held_out')
    text(4.6, 2.365, 'Held-out 10,000', panel=2)
    text(6.02, 2.49, 'Tests', panel=2, source=LABEL_SOURCE)
    for y, label in [(2.395, 'Transfer 71'), (2.31, 'Lookup 12'), (2.225, 'Clamp 3')]:
        text(6.02, y, label, panel=2)
    text(0.2, 1.5, 'Molecular', weight='bold', panel=3)
    text(2.39, 1.51, 'ESM-2 8M, block 3', ha='center', panel=3)
    if preview:
        strip(1.86, 1.245, 1.35, 0.1, n=9, selected=3, mask=True, role='protein_single_mask_residue_strip')
        for i, residue in enumerate('DLGAWWEER'):
            if i != 3:
                text(1.86 + (i + 0.5) * 0.15, 1.295, residue, size=7, ha='center', panel=3, role='protein_sequence_example', source='Frozen protein example: Q5SI12 / 2CZL chain A, 161–169, A164 masked')
    else:
        text(1.86, 1.27, '… A L', panel=3, role='protein_sequence_example', source='Frozen figure source')
        ax.add_patch(Rectangle((2.24, 1.225), 0.085, 0.105, facecolor=MASK, edgecolor=MASK, linewidth=0.5))
        text(2.4, 1.27, 'A …', panel=3, role='protein_sequence_example', source='Frozen figure source')
    text(2.39 if preview else 2.28, 1.14, 'masked residue', ha='center', panel=3)

    def program_pair(x, y, width, protein=False):
        for yy, augmented in [(y + 0.35, False), (y, True)]:
            box(x, yy, width, 0.245, fill='white', edge=BLUE, lw=0.95, radius=0.035)
            if protein:
                chip_w, chip_h = (0.19, 0.16)
                first = x + (0.045 if augmented else 0.12)
                for i, name in enumerate(['X', 'S', 'L']):
                    xx = first + i * 0.21
                    box(xx, yy + 0.0425, chip_w, chip_h, fill='white', edge=BLUE, lw=0.55, radius=0.015)
                    text(xx + chip_w / 2, yy + 0.122, name, ha='center', panel=3, role='fitted_program')
                if augmented:
                    text(x + 0.715, yy + 0.122, '+', ha='center', panel=3, role='fitted_program')
                    ax.add_patch(Rectangle((x + width - 0.16, yy + 0.073), 0.1, 0.1, facecolor=ORANGE, edgecolor='none'))
                glyphs.append({'role': 'protein_program_chips', 'names': ['X', 'S', 'L'], 'augmented': augmented, 'parent_bounds_in': [x, yy, width, 0.245], 'selected_unit_inside_right': augmented})
            elif augmented:
                text(x + width / 2 - 0.085, yy + 0.122, 'L +', ha='center', panel=3, role='fitted_program')
                ax.add_patch(Rectangle((x + width - 0.16, yy + 0.073), 0.1, 0.1, facecolor=ORANGE, edgecolor='none'))
            else:
                text(x + width / 2, yy + 0.122, 'L', ha='center', panel=3, role='fitted_program')
        endx = x + width + 0.31
        endy = y + 0.3
        arrow((x + width, y + 0.472), (endx, endy + 0.04), 'protein_without_unit_to_DeltaBrier' if protein else 'molecular_L_to_DeltaBrier')
        arrow((x + width, y + 0.122), (endx, endy - 0.04), 'protein_with_unit_to_DeltaBrier' if protein else 'molecular_L_plus_unit_to_DeltaBrier')
        text(endx + 0.04, endy, 'ΔBrier', ha='left', panel=3)
    if preview:
        text(0.18, 1.0, 'L', ha='center', panel=3, role='molecular_input', source='MANUSCRIPT.md:19; ' + DISPLAY_SOURCE + ':99-112')
        line([(0.28, 0.822), (0.28, 1.172)], lw=0.55)
        line([(0.22, 1.0), (0.28, 1.0)], lw=0.55)
        arrow((0.28, 1.172), (0.44, 1.172), 'molecular_input_to_L_program', lw=0.55)
        arrow((0.28, 0.822), (0.44, 0.822), 'molecular_input_to_L_plus_program', lw=0.55)
        program_pair(0.45, 0.7, 0.47)
    else:
        program_pair(0.22, 0.7, 0.58)
    program_pair(1.79, 0.48, 0.92, True)
    if preview:
        line([(1.7, 1.295), (1.855, 1.295)], lw=0.55)
    line([(1.7, 1.295 if preview else 1.27), (1.7, 0.602)], lw=0.55)
    arrow((1.7, 0.952), (1.78, 0.952), 'protein_input_to_program_without_unit', lw=0.55)
    arrow((1.7, 0.602), (1.78, 0.602), 'protein_input_to_program_with_unit', lw=0.55)
    line([(1.58, 0.48), (1.58, 1.55)], color='#CDD0D4', lw=0.6)
    box(0.16, 0.105 + LEGEND_DELTA, 6.68, 0.16, fill='white', edge='none', radius=0.025)
    box(0.24, 0.135 + LEGEND_DELTA, 0.11, 0.1, fill='white', edge=BLUE, lw=0.95, radius=0.015)
    text(0.41, 0.185 + LEGEND_DELTA, 'fitted program', panel=None, role='legend')
    ax.add_patch(Rectangle((1.73, 0.135 + LEGEND_DELTA), 0.1, 0.1, facecolor=ORANGE, edgecolor='none'))
    text(1.9, 0.185 + LEGEND_DELTA, 'selected unit', panel=None, role='legend')
    arrow((3.08, 0.185 + LEGEND_DELTA), (3.3, 0.185 + LEGEND_DELTA), 'legend_J1', color=J1, lw=1)
    text(3.37, 0.185 + LEGEND_DELTA, 'J1', panel=None, role='legend', color=J1)
    arrow((3.88, 0.185 + LEGEND_DELTA), (4.1, 0.185 + LEGEND_DELTA), 'legend_J2', color=J2, lw=1)
    text(4.17, 0.185 + LEGEND_DELTA, 'J2', panel=None, role='legend', color=J2)
    ax.add_patch(Rectangle((4.67, 0.135 + LEGEND_DELTA), 0.1, 0.1, facecolor=MASK, edgecolor='none'))
    text(4.84, 0.185 + LEGEND_DELTA, 'masked residue', panel=None, role='legend')
    glyphs.append({'role': 'unified_bottom_legend', 'bounds_in': [0.16, 0.105 + LEGEND_DELTA, 6.68, 0.16], 'entries': ['fitted program', 'selected unit', 'J1', 'J2', 'masked residue']})
    for y, label, color in [(1.18, 'M1\nbinary concept', BLUE), (0.69, 'M2\nlocal graph', TEAL)]:
        box(3.7, y, 0.67, 0.34, fill='white', edge=color, lw=1.0, radius=0.04)
        text(4.035, y + 0.17, label, ha='center', panel=4)
        arrow((4.38, y + 0.17), (4.54, 1.1 if y > 1 else 0.99), 'M1_to_predicted_activations' if y > 1 else 'M2_to_predicted_activations')
    box(4.55, 0.875, 0.65, 0.55, fill='#FFFFFF', edge='#A3ACB1', radius=0.04)
    text(4.875, 1.225, 'Predicted\nactivations', ha='center', panel=4)
    strip(4.61, 0.965, 0.53, 0.08, n=6, selected=1, role='predicted_activation_six_cell_strip')
    arrow((5.21, 0.96 if preview else 1.15), (5.35, 0.96) if preview else (5.29, 1.15), 'predicted_activations_to_replacement_tensor')
    gx, gy, cell = (5.36, 0.92, 0.095) if preview else (5.3, 0.92, 0.105)
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((gx + col * cell, gy + row * cell), cell, cell, facecolor=ORANGE if col == 2 else 'white', edgecolor='#9AA4AA', linewidth=0.5))
    glyphs.append({'role': 'replacement_tensor', 'rows': 4, 'columns': 4, 'orange_column': 2, 'bounds_in': [gx, gy, 4 * cell, 4 * cell], 'scientific_coordinate_count': 1})
    text(5.55 if preview else 5.51, 1.48, 'token ×\ncoordinate', ha='center', panel=4)
    if preview:
        text(5.28, 1.15, 'atoms', ha='center', rotation=90, panel=4, source='MANUSCRIPT.md:19; ' + DISPLAY_SOURCE + ':121')
    arrow((5.75 if preview else 5.73, 1.15), (5.84, 1.15), 'replacement_tensor_to_same_frozen_downstream')
    box(5.85, 0.88, 0.61, 0.55, fill='#FFFFFF', edge='#7F8F9B', radius=0.04)
    text(6.155, 1.075, 'Frozen\noutputs', ha='center', panel=4)
    box(6.1, 1.28, 0.11, 0.075, fill='none', edge=INK, lw=0.65, radius=0.01)
    lock_path = MplPath([(6.12, 1.355), (6.12, 1.405), (6.19, 1.405), (6.19, 1.355)], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(lock_path, facecolor='none', edgecolor=INK, linewidth=0.65))
    glyphs.append({'role': 'frozen_output_outline_lock', 'bounds_in': [6.1, 1.28, 0.11, 0.125]})
    arrow((6.47, 1.15), (6.6, 1.15), 'frozen_output_to_subtractor')
    ax.add_patch(Circle((6.7, 1.15), 0.085, facecolor='white', edgecolor=INK, linewidth=0.8))
    line([(6.66, 1.15), (6.74, 1.15)], lw=0.8)
    text(6.63, 1.6, 'Original\noutputs', ha='center', panel=4)
    arrow((6.7, 1.44), (6.7, 1.245), 'original_output_to_subtractor_from_above')
    arrow((6.7, 1.05), (6.7, 0.87), 'subtractor_to_residuals')
    text(6.64, 0.75, 'Residuals', ha='center', panel=4)
    box(5.175, 0.3, 0.67, 0.34, fill='white', edge=MUTED, lw=0.65, radius=0.035, linestyle='--')
    glyphs.append({'role': 'shuffled_control_box', 'bounds_in': [5.175, 0.3, 0.67, 0.34], 'matches_M2_width_height_in': [0.67, 0.34]})
    line([(5.76, 1.15), (5.76, 0.76), (5.51, 0.76)], style='--', color=MUTED, lw=0.65)
    arrow((5.51, 0.76), (5.51, 0.645), 'predicted_replacement_to_parallel_shuffle', style='--', color=MUTED, lw=0.65)
    line([(5.85, 0.47), (6.155, 0.47), (6.155, 0.76)], style='--', color=MUTED, lw=0.65)
    arrow((6.155, 0.76), (6.155, 0.875), 'shuffle_replacement_to_same_frozen_module', style='--', color=MUTED, lw=0.65)
    text(5.51, 0.47, 'Same-molecule\ndisplacement shuffle', ha='center', panel=4, color=MUTED)
    arrow((3.325, 3.91), (3.675, 3.91), 'navigation_A_to_B', lw=1.5, color='#555555')
    ux, uy = (0.13 / math.hypot(0.13, 0.12), 0.12 / math.hypot(0.13, 0.12))
    arrow((3.505 + 0.175 * ux, 2.1 + 0.175 * uy), (3.505 - 0.175 * ux, 2.1 - 0.175 * uy), 'navigation_B_to_C', lw=1.5, color='#555555')
    arrow((3.325, 1.79), (3.675, 1.79), 'navigation_C_to_D', lw=1.5, color='#555555')
    panel_bounds = [[x, y + BOTTOM_EXTENSION, w, h] for x, y, w, h in panel_bounds]
    for item in glyphs:
        for key in ['bounds_in', 'parent_bounds_in']:
            if key in item:
                item[key][1] += BOTTOM_EXTENSION
        if 'horizontal_divider_y_in' in item:
            item['horizontal_divider_y_in'] += BOTTOM_EXTENSION
    for item in structures:
        item['bounds_in'][1] += BOTTOM_EXTENSION
    geometry = {'canvas_in': [W, H], 'baseline_canvas_in': [W, BASE_H], 'baseline_version': 'round8_20261102_v02', 'bottom_extension_in': BOTTOM_EXTENSION, 'whole_figure_scaled': False, 'all_panel_content_translation_in': [0, BOTTOM_EXTENSION], 'all_panel_content_physical_dimensions_unchanged': True, 'distance_from_page_top_unchanged': True, 'panel_bounds_in': panel_bounds, 'four_panel_bottom_edge_y_in': min((p[1] for p in panel_bounds)), 'legend_strip_bounds_in': [0.16, LEGEND_BOTTOM, 6.68, LEGEND_HEIGHT], 'legend_strip_upper_edge_y_in': LEGEND_BOTTOM + LEGEND_HEIGHT, 'legend_net_gap_in': min((p[1] for p in panel_bounds)) - (LEGEND_BOTTOM + LEGEND_HEIGHT), 'legend_required_gap_in': LEGEND_GAP, 'minimum_font_pt': 7}
    assert abs(geometry['legend_net_gap_in'] - LEGEND_GAP) < 1e-12
    archive = OUTPUT_BASE / 'figure_1' / version
    archive.mkdir(parents=True, exist_ok=True)
    if (archive / 'figure_1.pdf').exists():
        raise RuntimeError('Choose a new output version')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    records = []
    for record in texts:
        rec = {k: v for k, v in record.items() if k != 'artist'}
        bb = record['artist'].get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        rec['ink_bounds_in'] = [float(v) for v in bb.extents]
        records.append(rec)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(str(archive / ('figure_1.' + ext)), dpi=600, facecolor='white')
    plt.close(fig)
    dump(archive / 'figure_1_geometry.json', geometry)
    return (archive, geometry)

def plot_chemical(g, left, bottom, fig, texts, panel, source):
    """Paint native point geometry; font remains independent, editable Arial 7 pt."""
    ax = fig.add_axes([left / W, (bottom + BOTTOM_EXTENSION) / H, g['width_pt'] / 72 / W, g['height_pt'] / 72 / H])
    ax.set_xlim(0, g['width_pt'])
    ax.set_ylim(g['height_pt'], 0)
    ax.set_aspect('equal')
    ax.axis('off')
    for a in g['atoms']:
        if a['index'] in g['matched_atom_indices']:
            ax.add_patch(Circle((a['x'], a['y']), 3.2, facecolor=ORANGE, edgecolor='none', alpha=0.25, zorder=1))
    atoms = {a['index']: a for a in g['atoms']}
    for path in g['paths']:
        ax.add_patch(PathPatch(MplPath(path['vertices'], path['codes']), facecolor='none', edgecolor=INK, linewidth=0.6, capstyle='round', joinstyle='round', zorder=2))
    for t in g['texts']:
        match = re.search('\\batom-(\\d+)\\b', t.get('source_class', ''))
        element = atoms[int(match.group(1))]['element'] if match else 'C'
        artist = ax.text(t['x'], t['y'], t['text'], fontsize=7, family='Arial', va='baseline', color=CPK.get(element, INK), ha={'start': 'left', 'middle': 'center', 'end': 'right'}.get(t.get('anchor'), 'left'), zorder=3)
        texts.append({'artist': artist, 'text': t['text'], 'size_pt': 7, 'source': source, 'role': 'chemical_atom_glyph', 'panel': panel, 'chemical_atom_index': int(match.group(1)) if match else None, 'anchor_in': [left + t['x'] / 72, bottom + BOTTOM_EXTENSION + (g['height_pt'] - t['y']) / 72]})
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--version', default='portable')
    args = p.parse_args()
    build(args.version, preview=True)
