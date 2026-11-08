"""Portable rendering from frozen values and geometry."""
"""Supplementary Figures S2 (all 18 Week 3 examples) and S3 (Week 3 null distributions for every layer).

usage: python -I figS2_S3.py
S2 tiles the example images made by scripts/lineB/examples_v4.py (lineB/results_v4/examples/), one page per model.
S3 recomputes J1 and J2 effects for every model x layer (all 173 concepts) with leave-one-out twin null clouds and
asserts equality with lineB/results_v4/gates_v4.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import historical_helpers as C
EX = C.ROOT / 'figure_source/assets/historical_examples'
ROWS = [('chemberta_mlm', 'chemberta', 'L2'), ('chemberta_mlm', 'chemberta', 'L1'), ('chemberta_mlm', 'chemberta', 'L0'), ('chemberta_mtr', 'chemberta', 'L2'), ('chemberta_mtr', 'chemberta', 'L1'), ('chemberta_mtr', 'chemberta', 'L0'), ('chemeleon', 'chemeleon', 'main'), ('chemeleon', 'chemeleon', 'mp3')]
ROW_LABEL = {'L2': 'layer 3 (main)', 'L1': 'layer 2', 'L0': 'layer 1', 'main': 'output (main)', 'mp3': 'after MP step 3'}

def fig_s2():
    ex = C.examples()
    for model, rows in ex.items():
        fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.6))
        for ax, r in zip(axes.ravel(), rows):
            ax.imshow(Image.open(f"{EX}/{r['figure']}").convert('RGB'))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{r['concept']}  |  unit {r['unit']} ({r['pole'][:3]}.)  |  lift {r['lift']:.0f}  |  precision@100 {r['precision_top100']:.2f}\nbest lift among 39 twins: median {r['twin_best_lift_median']:.1f}, max {r['twin_best_lift_max']:.1f}", fontsize=5.6, pad=2)
        fig.suptitle(f'{C.NAMES[model]}: six example concepts (held-out set H3, main layer)', fontsize=7.5, y=0.995)
        fig.subplots_adjust(hspace=0.25, wspace=0.04, top=0.94, bottom=0.01, left=0.01, right=0.99)
        C.save(fig, f'figS2_examples_{model}')
        plt.close(fig)

def fig_s3():
    g4 = json.loads((C.ROOT / 'source_data/figure_S_C5_gates.json').read_text(encoding='utf-8'))['models']
    res = []
    for m, group, layer in ROWS:
        d = C.v4_group(group, layer)
        r = C.joint_effects(d, m)
        ref = g4[m][layer]['all']
        C.check_against(r['E1'], ref['J1']['E_median_log_ratio'], f'{m} {layer} E1')
        C.check_against(r['E2'], ref['J2']['E_median_log_ratio'], f'{m} {layer} E2')
        res.append((m, layer, r, ref))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw=dict(wspace=0.55))
    y = np.arange(len(ROWS))[::-1]
    rng = np.random.default_rng(5)
    for ax, key, nkey, test, xl in ((axes[0], 'E1', 'null1', 'J1', 'J1: pretrained / untrained twins (fold)'), (axes[1], 'E2', 'null2', 'J2: native-vs-rotated advantage (fold)', None)):
        for yi, (m, layer, r, ref) in zip(y, res):
            ax.scatter(np.exp(r[nkey]), yi + rng.uniform(-0.18, 0.18, C.K), s=3, c=C.OI['grey'], alpha=0.7, linewidths=0)
            p = ref[test.split(':')[0]]['p']
            sig = p <= 0.05 and r[key] > 0
            ax.scatter([np.exp(r[key])], [yi], s=24, marker='D', zorder=3, edgecolors=C.MODEL_COLOR[m], facecolors=C.MODEL_COLOR[m] if sig else 'white', linewidths=1.0)
            ax.text(1.02, yi, f'{np.exp(r[key]):.2f}×  p={p:.3f}', transform=ax.get_yaxis_transform(), va='center', fontsize=5.6)
        ax.axvline(1, color='k', lw=0.5, ls=':')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(xl or test)
    axes[0].set_xticks([0.7, 1, 2, 5, 10])
    axes[1].set_xticks([0.9, 1, 1.5, 2])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'{C.NAMES[m]}, {ROW_LABEL[l]}' for m, l, _, _ in res])
    C.panel(axes[0], 'a', x=-0.95)
    C.panel(axes[1], 'b', x=-0.08)
    C.save(fig, 'figS3_week3_nulls')

def main():
    C.setup_style()
    fig_s2()
    fig_s3()
if __name__ == '__main__':
    main()
