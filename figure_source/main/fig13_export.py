"""Portable rendering from frozen values and geometry."""
import copy
import json
import shutil
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.text import Text
import paper_style as ps
from chemical_asset_style import draw_asset as draw_native
ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'figure_source/assets/figure_3'
PALETTE = {'N': '#0077BB', 'O': '#CC3311', 'S': '#CCBB44', 'Cl': '#009988', 'C': '#000000', 'R': '#000000', '*': '#000000', 'H': '#000000'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def frozen_geometry(name):
    path = FROZEN / (name + '_geometry.json')
    return (read(path), path)

def representative_assets():
    registry = {int(r['concept_id']): r for r in read(FROZEN / 'representative_smiles_registry.json')['entries']}
    result = {}
    for cid in [0, 14, 18, 24, 26]:
        g, path = frozen_geometry('fig4_concept_%d' % cid)
        result[cid] = dict(geometry=g, geometry_file=path.as_posix(), source_registry_entry=registry[cid])
    return result

def draw_asset(fig, left, bottom, g):
    ax = draw_native(fig, left, bottom, g)
    for halo in g.get('halos', []):
        ax.add_patch(Circle((halo['x'], halo['y']), halo['radius_pt'], facecolor=halo['color'], edgecolor='none', alpha=0.25, zorder=0.2))
    return ax

from fig12_helpers import save as save_number
def save_fixed(fig,number,version,data,meta):
    target,result=save_number(fig,number,version,data,meta['physical_layout'],Path(__file__))
    result.update(meta)
    result.update(width_in=float(fig.get_size_inches()[0]),height_in=float(fig.get_size_inches()[1]))
    return target,result
