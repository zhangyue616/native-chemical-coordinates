"""Portable rendering from frozen values and geometry."""
import argparse
import copy
import csv
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator
from fig12_helpers import ROOT, R7, BASE, ps, read, write, save, draw_asset
from glyphs import draw_state_matrix
P2 = ROOT / 'source_data/inputs/figure_2_AB.json'
P3 = ROOT / 'source_data/inputs/figure_2_CD.json'
W = H = 7.0
NAMES = ['ChemBERTa-77M-MLM', 'ChemBERTa-77M-MTR', 'CheMeleon', 'MolCLR GIN', 'Hu masking', 'Hu edgepred', 'Hu contextpred', 'Hu infomax', 'MoLFormer']
LABELS = ['ChemBERTa-\n77M-MLM', 'ChemBERTa-\n77M-MTR', 'CheMeleon', 'MolCLR GIN', 'Hu masking', 'Hu edgepred', 'Hu contextpred', 'Hu infomax', 'MoLFormer\n(layer 10)']
COLORS = {'J1': ps.CYAN, 'J2': ps.MAGENTA}

def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]

def text(fig, x, y, label, **kw):
    opts = dict(fontsize=7.5, va='center')
    opts.update(kw)
    return fig.text(x / W, y / H, label, **opts)

def tick(v, p=None):
    return format(float(v), '.3g')

def matrix(fig, left, center, cells):
    for ri, row in enumerate(cells):
        for ci, fill in enumerate(row):
            fig.patches.append(Rectangle(((left + ci * 0.12) / W, (center - ri * 0.075) / H), 0.12 / W, 0.075 / H, transform=fig.transFigure, facecolor=fill, edgecolor='#8C8C8C', linewidth=0.5))

def build(version):
    d2, d3 = (read(P2), read(P3))
    fig = ps.new_figure(W, H)
    mapping = copy.deepcopy(d2['source_display_records'])
    style = []
    audit, depth = (d2['audit'], d2['depth'])
    support = d2['support_matrix_records']
    text(fig, 0.1, 6.87, 'A  Checkpoint evidence', fontsize=9, weight='bold')
    handles = [Line2D([], [], marker='o', ls='None', ms=s, mfc=f, mec=e, mew=0.5, label=l) for s, f, e, l in [(6.5, ps.INK, ps.WHITE, 'Primary cohort, 199 twins'), (4.5, 'none', ps.INK, 'Earlier cohort, 39 twins')]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(4.37 / W, 6.88 / H), bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1)
    positions = list(range(8, -1, -1))
    bottom, height = (4.54, 1.98)
    offset = 0.18
    fy = bottom + (positions[1] + 0.5) / 9 * height
    fig.patches.append(Rectangle((0.15 / W, (fy - 0.11) / H), 6.64 / W, 0.22 / H, transform=fig.transFigure, facecolor=ps.ORANGE, alpha=0.1, edgecolor='none', zorder=-1))
    a_bounds = []
    glyphs = []
    for metric, left, limits, ticks in [('J1', 1.5, (0.55, 3.2), [0.6, 1, 2, 3]), ('J2', 3.85, (0.86, 2.25), [0.9, 1, 1.5, 2])]:
        bounds = [left, bottom, 2.1, height]
        a_bounds.append(bounds)
        ax = fig.add_axes(rect(*bounds))
        ax.set_xscale('log')
        ax.set_xlim(*limits)
        ax.set_ylim(-0.5, 8.5)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(tick))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks([])
        ps.clean_axes(ax)
        ax.set_facecolor('none')
        ax.spines['left'].set_visible(False)
        ax.axvline(1, color='#999999', lw=0.5, ls='--', zorder=0)
        for i, row in enumerate(audit['models']):
            earlier, primary = (row['H2'][metric + '_fold'], row['H4'][metric + '_fold'])
            ax.plot([earlier, primary], [positions[i] + offset, positions[i] - offset], color=COLORS[metric], alpha=0.5, lw=1)
            for val, y, fill, edge, size in [(earlier, positions[i] + offset, 'none', COLORS[metric], 4.5), (primary, positions[i] - offset, COLORS[metric], ps.WHITE, 6.5)]:
                ax.plot(val, y, 'o', ms=size, mfc=fill, mec=edge, mew=0.5)
        ax.plot(audit['molformer_primary'][metric]['fold'], 0, 'o', ms=6.5, mfc=COLORS[metric], mec=ps.WHITE, mew=0.5)
        ax.set_xlabel(metric + ' fold (log scale)', fontsize=8, labelpad=2)
        text(fig, left + 0.73, 6.63, metric, fontsize=9, weight='bold', color=COLORS[metric])
        glyphs.append(draw_state_matrix(fig, left + 1.06, 6.55, 0.16, metric=metric))
    for i, (y, label, row) in enumerate(zip(positions, LABELS, support)):
        fy = bottom + (y + 0.5) / 9 * height
        text(fig, 1.37, fy, label, ha='right', linespacing=0.96, weight='bold' if i == 1 else 'normal')
        matrix(fig, 6.36, bottom + (y - (offset if i < 8 else 0) + 0.5) / 9 * height, row['cells'])
    sep = bottom + (2.049 - 1.845)
    fig.lines.append(Line2D([1.45 / W, 6.84 / W], [sep / H] * 2, transform=fig.transFigure, color=ps.GRAY, lw=0.5))
    text(fig, 6.48, 6.87, 'Support', fontsize=8, weight='bold', ha='center')
    text(fig, 6.48, 6.72, 'Primary cohort', ha='center')
    for x, s in [(6.42, 'J1'), (6.54, 'J2')]:
        text(fig, x, 6.57, s, ha='center', color=COLORS[s])
    for x, label, color in [(0.1, '', ps.CYAN), (0.21, 'Pass (J1 / J2)', ps.MAGENTA), (1.75, 'Nominal-only', ps.GRAY), (3.04, 'Not met', ps.WHITE)]:
        fig.patches.append(Rectangle((x / W, 4.13 / H), 0.08 / W, 0.08 / H, transform=fig.transFigure, facecolor=color, edgecolor='#8C8C8C', lw=0.5))
        text(fig, x + 0.13, 4.17, label)
    text(fig, 3.96, 4.17, 'Direction / Calibrated', fontsize=7.5)
    text(fig, 0.1, 3.99, 'B  MoLFormer depth profile', fontsize=9, weight='bold')
    b_bounds = []
    for metric, left, limits, ticks in [('J1', 0.99, (0.88, 2.32), [1, 1.5, 2]), ('J2', 4.34, (0.96, 1.42), [1, 1.2, 1.4])]:
        bounds = [left, 3.1, 2.3, 0.75]
        b_bounds.append(bounds)
        ax = fig.add_axes(rect(*bounds))
        series = next((r['values'] for r in depth if r['metric'] == metric))
        ax.plot(range(1, 13), series, color=COLORS[metric], marker='o', ms=6.5, lw=1.25, mec=ps.WHITE, mew=0.5)
        ax.plot(10, series[9], 'o', mfc=ps.ORANGE, mec=ps.WHITE, ms=6.5, mew=0.5, zorder=4)
        ax.axhline(1, color='#999999', lw=0.5, ls='--', zorder=0)
        ax.set_xlim(0.75, 12.25)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(tick))
        ax.set_xticks(range(1, 13))
        ax.set_xlabel('Layer', fontsize=8, labelpad=2)
        ax.set_ylabel(metric + ' fold\n(linear scale)', fontsize=7.5, rotation=0, ha='right', va='center', labelpad=5, color=ps.INK)
        ps.clean_axes(ax)
        if metric == 'J1':
            text(fig, left + 1.5, 3.99, 'J1-selected layer 10', color=ps.ORANGE, ha='center')
    text(fig, 0.1, 2.61, 'C  Ranked per-concept transfer', fontsize=9, weight='bold')
    text(fig, 4.75, 2.71, 'D  Joint raw transfer', fontsize=9, weight='bold')
    c_bounds = [0.56, 0.46, 3.69, 1.45]
    d_bounds = [4.75, 0.4, 2.2, 2.2]
    axc = fig.add_axes(rect(*c_bounds))
    axd = fig.add_axes(rect(*d_bounds))
    names = {int(r['id']): r for r in d3['approved_names']}
    raw = {int(r['exact_id']): r for r in d3['source_records']['raw']}
    scaffold = {int(r['concept_id']): r for r in d3['source_records']['scaffold']}
    order = [r['id'] for r in sorted(d3['approved_names'], key=lambda r: r['raw_theta_rank'])]
    lookup = set(d3['frozen_selection_record']['panel_ids'])
    clamp = set(d3['frozen_selection_record']['mechanism_candidate_ids'])
    ys = [float(raw[c]['difference_Btrained_minus_BinitMedian']) for c in order] + [float(scaffold[c]['Theta_point']) for c in order]
    low, high = (min(min(ys), 0), max(max(ys), 0))
    pad = (high - low) * 0.1
    axc.set_xlim(0.1, 71.9)
    axc.set_ylim(low - pad, high + pad)
    axc.set_xticks([1, 10, 20, 30, 40, 50, 60, 71])
    axc.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axc.yaxis.set_major_formatter(FuncFormatter(lambda v, p: format(v, '.2g')))
    axc.set_xlabel('Rank by raw held-out Θ', fontsize=8, labelpad=2)
    axc.set_ylabel('Log-lift contrast Θ', fontsize=8, labelpad=2)
    ps.clean_axes(axc)
    axc.axhline(0, color='#999999', lw=0.5, ls='--', zorder=0)
    means = copy.deepcopy(d3['mean_lines'])
    for rec, label, sty, label_y, color in [(means[0], 'Raw held-out mean', '-', 1.65, ps.DARK_GRAY), (means[1], 'Scaffold-unseen mean', '--', 1.17, ps.CYAN)]:
        axc.axhline(rec['source_value'], color=color, lw=0.5, ls=sty)
        axc.annotate(label, (71.9, rec['source_value']), xytext=(70.8, label_y), fontsize=7.5, ha='right', va='center', color=color, arrowprops=dict(arrowstyle='-', color=color, lw=0.5, shrinkA=2, shrinkB=0))
    for c in order:
        rank = names[c]['raw_theta_rank']
        for cohort, marker, y in [('raw', 'o', float(raw[c]['difference_Btrained_minus_BinitMedian'])), ('scaffold', '^', float(scaffold[c]['Theta_point']))]:
            color = ps.ORANGE if c in clamp else ps.INK if c in lookup else ps.DARK_GRAY if cohort == 'raw' else ps.CYAN
            if c in clamp:
                focal = c == 24
                axc.plot(rank, y, marker='s', ms=4, ls='none', mfc=ps.ORANGE if focal else ps.WHITE, mec=ps.WHITE if focal else ps.ORANGE, mew=0.5, zorder=4)
                axc.plot(rank, y, marker=marker, ms=2.6, ls='none', mfc='none' if focal or cohort == 'scaffold' else ps.ORANGE, mec=ps.WHITE if focal else ps.ORANGE, mew=0.45, zorder=5)
            else:
                axc.plot(rank, y, marker=marker, ms=4, ls='none', mfc=color if cohort == 'raw' else 'none', mec=ps.WHITE if cohort == 'raw' else color, mew=0.5, zorder=3 if c in lookup else 2)
            style.append(dict(panel='C', concept_id=c, cohort=cohort, rank=rank, source_y=y, marker_size_pt=4, color=color, cohort_marker=marker, membership_square=c in clamp))
    for c, pos, label in [(24, (25, 1.23), 'Carbonyl'), (25, (40, 1.72), 'Carbonyl (noCOO)'), (26, (50, 0.82), 'Aliphatic OH')]:
        axc.annotate(label, (names[c]['raw_theta_rank'], float(raw[c]['difference_Btrained_minus_BinitMedian'])), xytext=pos, fontsize=7.5, ha='center', va='center', arrowprops=dict(arrowstyle='-', color=ps.DARK_GRAY, lw=0.5, shrinkA=2, shrinkB=3))
    fragment_records = []
    for c, center in [(18, 0.94), (38, 1.68), (90, 2.58)]:
        p = ROOT / 'figure_source/assets/figure_2' / ('fig3_concept_%d_geometry.json' % c)
        g = read(p)
        height = g['height_pt'] / 72
        width = g['width_pt'] / 72
        bottom = 2.18 - height / 2
        draw_asset(fig, center - width / 2, bottom, g)
        text(fig, center - width / 2 - 0.06, 2.18, str(names[c]['raw_theta_rank']), ha='right')
        rank = names[c]['raw_theta_rank']
        target = float(raw[c]['difference_Btrained_minus_BinitMedian'])
        axc.annotate('', xy=(rank, target), xytext=(center / W, (bottom - 0.025) / H), textcoords='figure fraction', arrowprops=dict(arrowstyle='-', color=ps.DARK_GRAY, lw=0.5, shrinkA=0, shrinkB=3), annotation_clip=False)
        fragment_records.append(dict(concept_id=c, geometry_file=str(p), rank=rank, leader_target_data=[rank, target], native_size_in=[width, height]))
    handles = [Line2D([], [], marker=m, ls='none', ms=4, mfc=f, mec=e, mew=0.5, label=l) for m, f, e, l in [('o', ps.DARK_GRAY, ps.WHITE, 'Raw held-out'), ('^', 'none', ps.CYAN, 'Scaffold-unseen')]]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.56 / W, 2.48 / H), bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=0.7, borderaxespad=0)
    categories = [('other', ps.GRAY, 'o', 4.5, 'Other concepts'), ('lookup_only', ps.INK, 'o', 6.5, 'Lookup only'), ('clamping', ps.ORANGE, 's', 6.5, 'Clamping'), ('focal_carbonyl', ps.ORANGE, 's', 6.5, 'Focal carbonyl')]

    def membership(c):
        return 'focal_carbonyl' if c == 24 else 'clamping' if c in clamp else 'lookup_only' if c in lookup else 'other'
    xs = [float(raw[c]['difference_Btrained_minus_BinitMedian']) for c in raw]
    ys = [float(raw[c]['Gconcept']) for c in raw]
    x0, x1 = (min(min(xs), 0), max(max(xs), 0))
    y0, y1 = (min(min(ys), 0), max(max(ys), 0))
    axd.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
    axd.set_ylim(y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    ps.clean_axes(axd)
    axd.axvline(0, color='#999999', lw=0.5, ls='--')
    axd.axhline(0, color='#999999', lw=0.5, ls='--')
    for rec in means[2:]:
        (axd.axvline if rec['orientation'] == 'vertical' else axd.axhline)(rec['source_value'], color='#999999', lw=0.5)
    axd.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axd.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axd.xaxis.set_major_formatter(FuncFormatter(lambda v, p: format(v, '.2g')))
    axd.yaxis.set_major_formatter(FuncFormatter(lambda v, p: format(v, '.2g')))
    axd.set_xlabel('Raw Θ (held-out)', fontsize=8, labelpad=2)
    d_label = [text(fig, 4.32, 1.38, 'G', fontsize=8, rotation=90, rotation_mode='anchor'), text(fig, 4.35, 1.465, 'init', fontsize=7.5, rotation=90, rotation_mode='anchor')]
    for cat, color, marker, size, label in categories:
        for c in raw:
            if membership(c) == cat:
                x, y = (float(raw[c]['difference_Btrained_minus_BinitMedian']), float(raw[c]['Gconcept']))
                axd.plot(x, y, marker=marker, ls='none', ms=size, mfc='none' if cat == 'clamping' else color, mec=color if cat == 'clamping' else ps.WHITE, mew=0.5, zorder=4 if c in lookup else 2)
                style.append(dict(panel='D', concept_id=c, category=cat, source_x=x, source_y=y, color=color, marker=marker, marker_size_pt=size, hollow=cat == 'clamping'))
    for c, pos, label in [(24, (1.48, 0.65), 'Carbonyl'), (25, (1.78, 2.85), 'Carbonyl\n(noCOO)'), (26, (1.45, 4.32), 'Aliphatic OH')]:
        axd.annotate(label, (float(raw[c]['difference_Btrained_minus_BinitMedian']), float(raw[c]['Gconcept'])), xytext=pos, fontsize=7.5, ha='center', va='center', arrowprops=dict(arrowstyle='-', color=ps.DARK_GRAY, lw=0.5, shrinkA=2, shrinkB=3))
    handles = [Line2D([], [], marker=m, ms=s, ls='none', mfc='none' if cat == 'clamping' else color, mec=color if cat == 'clamping' else ps.WHITE, mew=0.5, label=label) for cat, color, m, s, label in categories]
    leg = axd.legend(handles=handles, loc='lower right', fontsize=7.5, frameon=True, borderpad=0, handletextpad=0.25, labelspacing=0.1, handlelength=0.7, borderaxespad=0.1)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('white')
    leg.get_frame().set_alpha(1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = [t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()) for t in d_label]
    center_y = (min((b.y0 for b in label_boxes)) + max((b.y1 for b in label_boxes))) / 2
    for t in d_label:
        x, y = t.get_position()
        t.set_position((x, y + (1.5 - center_y) / H))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    d_label_records = []
    for t in d_label:
        bb = t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        d_label_records.append({'text': t.get_text(), 'anchor_in': [t.get_position()[0] * W, t.get_position()[1] * H], 'rotation': t.get_rotation(), 'fontsize_pt': t.get_fontsize(), 'color': t.get_color(), 'ink_bounds_in': list(map(float, bb.extents))})
    geometry = {'canvas_in': [7, 7], 'requested_row_heights_in': [2.4, 1.4, 2.4], 'row_content_bands_in': {'A': [4.5, 6.9], 'B': [2.75, 4.15], 'CD': [0.3, 2.7]}, 'remaining_vertical_budget_in': 0.8, 'A_axes_in': a_bounds, 'B_axes_in': b_bounds, 'C_panel_width_in': 4.6, 'C_axes_in': c_bounds, 'D_axes_in': d_bounds, 'D_data_axes_square_in': 2.2, 'C_D_panel_gap_in': 0.15, 'C_markers_pt': 4, 'C_nested_cohort_marker_pt': 2.6, 'D_y_label_inside_axes': False, 'D_y_label_location': 'left of y ticks, centered at y=1.50in', 'D_y_label_reason': 'Frozen figure source', 'D_y_label': {'semantic_text': 'G_init', 'rotation': 90, 'group_center_y_in': 1.5, 'D_y_axis_midpoint_in': 0.4 + 2.2 / 2, 'editable_components': d_label_records}, 'B_y_label_colors': {'J1': '#000000', 'J2': '#000000'}, 'baseline_version': 'r8_20261102_v04', 'A_checkpoint_row_spacing_in': 0.22, 'A_eight_checkpoint_MoLFormer_separator': {'x_bounds_in': [1.45, 6.84], 'y_in': sep, 'color': ps.GRAY, 'linewidth_pt': 0.5, 'frozen_R7_offset_from_data_bottom_in': 2.049 - 1.845}, 'C_representative_fragments': fragment_records, 'decorative_state_matrices': glyphs, 'Support_statistics_distinct_from_decorative_state_matrices': True, 'minimum_font_pt': 7, 'deleted_visible_strings': ['Membership overrides hue', 'Atom key (CPK): N'], 'point_counts': {'A': 34, 'B': 24, 'C': 142, 'D': 71}, 'whole_figure_scaled': False}
    data = {'figure': 2, 'version': version, 'sources': {'R7_figure2': str(P2), 'R7_figure3': str(P3)}, 'all_scientific_values_unchanged': True, 'new_scientific_computation': False, 'panel_AB_frozen_source': d2, 'panel_CD_frozen_source': d3, 'source_display_records_AB': mapping, 'style_records_CD': style, 'physical_layout': geometry}
    target, meta = save(fig, 2, version, data, geometry, Path(__file__))
    print(str(target))
    print(meta['PNG_pixel_size'])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--version', default='portable')
    build(p.parse_args().version)
