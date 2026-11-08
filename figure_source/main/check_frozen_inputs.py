"""Import renderers and load their frozen inputs without drawing or estimating."""
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]


def main():
    modules = [importlib.import_module('plot_figure_%d' % i) for i in range(1, 5)]
    one, two, three, four = modules
    evidence = json.loads(one.TOKEN_EVIDENCE.read_text(encoding='utf-8'))
    assert evidence['tokens'] == ['*', 'C', '(', '=', 'O', ')', '*']
    for stem in ['T1_CID60628745', 'T2_CID43411971', 'T3_CID678545']:
        candidates = json.loads((one.ASSETS / (stem + '_layout_candidates.json')).read_text(encoding='utf-8'))
        assert candidates and all('atoms' in row and 'paths' in row for row in candidates)
    ab, cd = two.read(two.P2), two.read(two.P3)
    assert len(ab['audit']['models']) == 8 and len(cd['approved_names']) == 71
    for cid in [18, 38, 90]:
        two.read(ROOT / 'figure_source/assets/figure_2' / ('fig3_concept_%d_geometry.json' % cid))
    baseline = three.read(three.SOURCE)
    assert len(baseline['concept_rows']) == 12 and len(baseline['display_values']) == 72
    assert len(three.representative_assets()) == 5
    three.frozen_geometry('T1_CID60628745')
    replacement, selected = four.read(four.SOURCE), four.read(four.SELECTION)
    assert len(selected['molecules']) == 2
    assert replacement['activation_rows'] and replacement['output_rows']
    for item in selected['molecules']:
        four.read(four.FROZEN / ('molecule_%s_raw_geometry.json' % item['molecule_id']))
    manifest = json.loads((ROOT / 'figure_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['figures']) == 15
    for item in manifest['figures']:
        assert (ROOT / item['png']).is_file() and (ROOT / item['source_data']).is_file()
    print(json.dumps({'imported_main_renderers': 4, 'manifest_figures': 15,
                      'frozen_input_loading': 'passed', 'figures_drawn': 0,
                      'new_scientific_estimates': 0}))


if __name__ == '__main__':
    main()
