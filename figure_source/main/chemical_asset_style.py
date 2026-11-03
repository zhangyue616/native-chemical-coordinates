"""Portable rendering from frozen values and geometry."""
"""Thin RDKit geometry -> editable native-point chemical assets.

RDKit preparation uses .venv_chem; the existing Anaconda runtime exports and
places the saved point geometry without importing RDKit. No asset is scaled.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
STYLE = {'atom_center_bond_length_pt': 11.5, 'black_bond_linewidth_pt': 0.6, 'atom_font_family': 'Arial', 'atom_fontsize_pt': 7.0, 'match_circle_linewidth_pt': 0.9, 'match_circle_radius_pt': 3.2, 'match_circle_fill': 'none', 'native_ink_margin_pt': 1.0, 'PNG_dpi': 600, 'whole_asset_scaling_allowed': False, 'atom_label_alignment': 'Actual Arial7pt primary chemical-symbol ink centered at atom; annotations follow rigidly', 'source_RDKit_SVG_pixel_font_is_not_final_physical_font': True}

def _paint(ax, geometry, offset_x=0, offset_y=0):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch, Circle
    for path in geometry['paths']:
        vertices = [[x + offset_x, y + offset_y] for x, y in path['vertices']]
        patch = PathPatch(MplPath(vertices, path['codes']), facecolor='none', edgecolor=path.get('color', 'black'), linewidth=path.get('linewidth_pt', 0.6), capstyle='round', joinstyle='round')
        if path.get('dash'):
            dash = [float(v) * geometry['source_SVG_to_point_unit'] for v in re.findall('\\d*\\.?\\d+', path['dash'])]
            patch.set_linestyle((0, dash))
        ax.add_patch(patch)
    for label in geometry['texts']:
        ax.text(label['x'] + offset_x, label['y'] + offset_y, label['text'], fontsize=7, family='Arial', color=label.get('color', 'black'), weight=label.get('font_weight', 'normal'), va='baseline', ha={'start': 'left', 'middle': 'center', 'end': 'right'}.get(label['anchor'], 'left'))
    for circle in geometry['circles']:
        ax.add_patch(Circle((circle['x'] + offset_x, circle['y'] + offset_y), circle.get('radius_pt', STYLE['match_circle_radius_pt']), fill=False, edgecolor=circle.get('color', '#000000'), linewidth=circle.get('linewidth_pt', 0.9)))

def draw_asset(fig, left_in, bottom_in, geometry_or_json_path):
    """Place an already finalized asset at its exact native physical size."""
    geometry = json.loads(Path(geometry_or_json_path).read_text(encoding='utf-8')) if isinstance(geometry_or_json_path, (str, Path)) else geometry_or_json_path
    if not geometry['geometry_finalized']:
        raise ValueError('Use finalized native geometry from create_asset before placement')
    fw, fh = fig.get_size_inches()
    width, height = (geometry['width_pt'], geometry['height_pt'])
    ax = fig.add_axes([left_in / fw, bottom_in / fh, width / 72 / fw, height / 72 / fh])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    _paint(ax, geometry)
    ax._chemical_asset_audit = {'native_size_in': [width / 72, height / 72], 'scaling_applied': False, 'style': geometry['style']}
    return ax

def _align_atom_text(geometry):
    """Correct RDKit-font baseline assumptions using actual Arial ink metrics.

    Only explicit atom classes/indices are eligible. Derived context notes do
    not acquire atom identity just because their text mentions C, O or N.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath, TextToPath
    text_to_path = TextToPath()
    groups = {}
    for index, label in enumerate(geometry['texts']):
        atom_index = label.get('atom_index')
        if atom_index is None:
            match = re.search('\\batom-(\\d+)\\b', label.get('source_class', ''))
            atom_index = int(match.group(1)) if match else None
        if atom_index is not None:
            groups.setdefault(int(atom_index), []).append(index)

    def ink_bounds(label):
        prop = FontProperties(family='Arial', size=7, weight=label.get('font_weight', 'normal'))
        bbox = TextPath((0, 0), label['text'], size=7, prop=prop).get_extents()
        advance = text_to_path.get_text_width_height_descent(label['text'], prop, False)[0]
        anchor = label.get('anchor', 'start')
        origin_x = label['x'] - (advance / 2 if anchor == 'middle' else advance if anchor == 'end' else 0)
        return [float(origin_x + bbox.x0), float(label['y'] - bbox.y1), float(origin_x + bbox.x1), float(label['y'] - bbox.y0)]
    records = []
    atoms = {int(atom['index']): atom for atom in geometry['atoms']}
    for atom_index, indices in groups.items():
        atom = atoms[atom_index]
        main = [i for i in indices if geometry['texts'][i].get('alignment_role') == 'atom_primary']
        symbol = atom['element']
        if not main:
            choices = [symbol.lower(), symbol] if atom.get('is_aromatic') else [symbol]
            for choice in choices:
                exact = [i for i in indices if geometry['texts'][i]['text'] == choice]
                if exact:
                    main = exact[:1]
                    break
                for start in range(len(indices)):
                    combined, selected = ('', [])
                    for i in indices[start:]:
                        combined += geometry['texts'][i]['text']
                        selected.append(i)
                        if combined == choice:
                            main = selected
                            break
                        if len(combined) >= len(choice):
                            break
                    if main:
                        break
                if main:
                    break
        if not main:
            main = [i for i in indices if re.match('^[A-Za-z]+$', geometry['texts'][i]['text']) and geometry['texts'][i]['text'] != 'H'][:1]
        if not main:
            continue
        boxes = [ink_bounds(geometry['texts'][i]) for i in main]
        center_x = (min((b[0] for b in boxes)) + max((b[2] for b in boxes))) / 2
        center_y = (min((b[1] for b in boxes)) + max((b[3] for b in boxes))) / 2
        dx, dy = (float(atom['x'] - center_x), float(atom['y'] - center_y))
        for index in indices:
            geometry['texts'][index]['x'] += dx
            geometry['texts'][index]['y'] += dy
        records.append({'atom_index': atom_index, 'primary_symbol': ''.join((geometry['texts'][i]['text'] for i in main)), 'text_component_indices': indices, 'translation_pt': [dx, dy], 'primary_ink_center_relative_to_atom_after_pt': [0.0, 0.0]})
    geometry['atom_label_alignment_records'] = records
    geometry['atom_label_alignment_method'] = 'Actual Arial7pt TextPath ink bounds, preserving per-atom component baseline differences; no atom/bond/circle movement'
