"""Portable rendering from frozen values and geometry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import csv
import math
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import paper_style as ps
from glyphs import draw_radius1, draw_sequence_window
from fig13_export import draw_asset, representative_assets, frozen_geometry, PALETTE, save_fixed, read, write
ROOT = Path(__file__).resolve().parents[2]
BASE = ps.BASE
SOURCE = ROOT / 'source_data/inputs/figure_3_baseline.json'
GROUPS = [('Rings', [0, 1, 2, 3]), ('Carboxylic acids', [14, 15, 16]), ('Carbonyls', [24, 25]), ('Hydroxyls', [26, 27]), ('Amine', [18])]
W, H = (7.4, 6.0)
FAMILIES = {'Rings': (ps.CYAN, 'o'), 'Carboxylic acids': (ps.RED, '^'), 'Carbonyls': (ps.ORANGE, 's'), 'Hydroxyls': (ps.MAGENTA, 'v'), 'Amine': (ps.DARK_GRAY, 'P')}

def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]

def axis(fig, x, y, w, h):
    return fig.add_axes(rect(x, y, w, h))

def text(fig, x, y, label, **kwargs):
    opt = dict(fontsize=7, ha='left', va='center')
    opt.update(kwargs)
    return fig.text(x / W, y / H, label, **opt)

def secondary_icon(fig, left, cy, kind):
    w, h = (17.28, 8.64)
    ax = axis(fig, left, cy - h / 144, w / 72, h / 72)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    if kind == 'HELIX':
        xs = [0.5 + (w - 1) * i / 384 for i in range(385)]
        ys = [h / 2 + (h - 1) / 2 * math.sin(6 * math.pi * i / 384) for i in range(385)]
        ax.plot(xs, ys, color='black', linewidth=1, solid_capstyle='round', clip_on=False)
    else:
        ax.add_patch(Polygon([[0, 2.82], [10.4, 2.82], [10.4, 0], [w, h / 2], [10.4, h], [10.4, 5.82], [0, 5.82]], closed=True, facecolor='black', edgecolor='none'))
    return {'kind': kind, 'native_width_pt': w, 'native_height_pt': h, 'left_in': left, 'center_y_in': cy}

def group_layout(geometries):
    cursor = 0.86
    rows, headers, separators = ({}, {}, [])
    for group, ids in reversed(GROUPS):
        for cid in reversed(ids):
            rows[cid] = cursor
            cursor += 0.22
        cursor -= 0.22
        half = geometries[ids[0]]['height_pt'] / 144
        center = cursor + 0.03 + 0.03 + half
        headers[group] = center
        cursor = center + max(half, 0.055)
        if group != 'Rings':
            separator = cursor + 0.08
            separators.append(separator)
            cursor = separator + 0.08 + 0.055
    return (rows, headers, separators)

def main(version):
    baseline = read(SOURCE)
    rows = copy.deepcopy(baseline['concept_rows'])
    by_id = {r['concept_id']: r for r in rows}
    assert [r['concept_id'] for r in rows] == [cid for _, ids in GROUPS for cid in ids]
    values = copy.deepcopy(baseline['display_values'])
    assert len(values) == 72
    cpk = representative_assets()
    registry = {cid: item['source_registry_entry'] for cid, item in cpk.items()}
    group_assets = {cid: item['geometry'] for cid, item in cpk.items()}
    row_positions, headers, separators = group_layout(group_assets)
    losses = {r['model']: r for r in baseline['absolute_losses']}
    endpoints = {r['endpoint']: r for r in baseline['endpoints']}
    example = copy.deepcopy(baseline['sequence_example'])
    sequence = example['sequence_fragment']
    assert len(sequence) == 22
    fig = ps.new_figure(W, H)
    panels = {'A': [0.15, 0.15, 3.9, 5.7], 'B': [4.2, 3.75, 3.05, 2.1], 'C': [4.2, 2.1, 3.05, 1.5], 'D': [4.2, 0.15, 3.05, 1.8]}
    text(fig, 1.46, 5.85, 'A  Molecular lookup increment', fontsize=9, weight='bold', va='top')
    radius_glyph = draw_radius1(fig, 3.84, 5.68, 0.2)
    text(fig, 4.2, 5.85, 'B  Frozen protein programs', fontsize=9, weight='bold', va='top')
    fig.lines.append(Line2D([4.125 / W, 4.125 / W], [0.15 / H, 5.85 / H], transform=fig.transFigure, color=ps.GRAY, linewidth=0.5))
    molecule, molecule_path = frozen_geometry('T1_CID60628745')
    mol_ax = draw_asset(fig, 0.15, 4.95, molecule)
    local = axis(fig, 1.46, 5.13, 2.54, 0.52)
    local.set_xlim(0, 1)
    local.set_ylim(0, 1)
    local.set_axis_off()
    for yy, marker, color, label in [(0.87, 'o', ps.ORANGE, 'Selected carbonyl'), (0.19, 'D', ps.DARK_GRAY, 'Panel mean (95% CI)')]:
        local.plot(0.025, yy, marker=marker, color=color, markersize=6.5, markerfacecolor='white' if marker == 'D' else color, markeredgecolor='black' if color == ps.ORANGE else color if marker == 'D' else 'white', markeredgewidth=0.5, linestyle='None', clip_on=False)
        local.text(0.09, yy, label, fontsize=7, va='center')
    plot = axis(fig, 1.62, 0.4, 1.04, 4.38)
    plot.set_xlim(-0.00025, 0.0041)
    plot.set_ylim(0.4, 4.78)
    plot.set_xticks([0, 0.001, 0.002, 0.003, 0.004])
    plot.set_xticklabels(['0', '1', '2', '3', '4'])
    plot.set_yticks([])
    ps.clean_axes(plot)
    plot.spines['left'].set_visible(False)
    plot.set_facecolor('none')
    plot.axvline(0, color='#999999', linewidth=0.5, linestyle='--', zorder=0)
    plot.tick_params(labelsize=7)
    plot.set_xlabel('ΔBrier (0.001 units)', fontsize=8, labelpad=4)
    labels = axis(fig, 0.15, 0.4, 1.4, 4.38)
    labels.set_xlim(0, 1)
    labels.set_ylim(0.4, 4.78)
    labels.set_axis_off()
    ledger = axis(fig, 2.77, 0.4, 1.28, 4.38)
    ledger.set_xlim(0, 1.28)
    ledger.set_ylim(0.4, 4.78)
    ledger.set_axis_off()
    colx = {'Native_AP': 0.21, 'Lookup_AP': 0.69, 'Delta_Brier': 1.16}
    for field, label in [('Native_AP', 'Native\nAP'), ('Lookup_AP', 'Lookup\nAP'), ('Delta_Brier', 'ΔBrier\n× 0.001')]:
        text(fig, 2.77 + colx[field], 4.85, label, fontsize=7, weight='bold', ha='center')
    rendered = {(r['concept_id'], r['field']): r['display'] for r in values if r['panel'] == 'A' and r['concept_id'] != ''}
    placements = []
    for group, ids in GROUPS:
        asset = group_assets[ids[0]]
        center = headers[group]
        draw_asset(fig, 0.15, center - asset['height_pt'] / 144, asset)
        family_color, family_marker = FAMILIES[group]
        text(fig, 0.15 + asset['width_pt'] / 72 + 0.065, center, group.replace('Carboxylic acids', 'Carboxylic\nacids'), fontsize=8, weight='bold', color=family_color)
        placements.append({'group': group, 'representative_concept_id': ids[0], 'geometry_file': cpk[ids[0]]['geometry_file'], 'exact_input_SMILES': registry[ids[0]]['exact_input_SMILES'], 'R_mapping': registry[ids[0]]['R_mapping'], 'center_y_in': center, 'native_size_in': [asset['width_pt'] / 72, asset['height_pt'] / 72], 'query_circles': 0, 'scaled': False, 'first_concept_example_not_group_definition': True})
        for cid in ids:
            row = by_id[cid]
            y = row_positions[cid]
            if cid == 24:
                fig.patches.append(Rectangle((0.15 / W, (y - 0.1) / H), 3.9 / W, 0.2 / H, transform=fig.transFigure, facecolor=ps.ORANGE, alpha=0.1, edgecolor='none', zorder=-1))
            labels.text(1, y, row['display_name'], fontsize=7, ha='right', va='center', weight='bold' if cid == 24 else 'normal')
            color = ps.ORANGE if cid in [24, 25] else family_color
            marker = 'o' if cid in [24, 25] else family_marker
            plot.plot(row['Delta_Brier'], y, marker=marker, linestyle='None', markerfacecolor=ps.WHITE if cid == 25 else color, markeredgecolor=ps.INK if cid == 24 else color if cid == 25 else ps.WHITE, markeredgewidth=0.5, markersize=6.5)
            for field, left in [('Native_AP', 0.0), ('Lookup_AP', 0.48)]:
                value = float(row[field])
                bar_width = 0.17
                ledger.add_patch(Rectangle((left, y - 0.04), bar_width, 0.08, facecolor='white', edgecolor='#DDDDDD', linewidth=0.5))
                ledger.add_patch(Rectangle((left, y - 0.04), bar_width * value, 0.08, facecolor='#DDDDDD', edgecolor='none'))
                ledger.text(left + bar_width + 0.025, y, rendered[cid, field], fontsize=7, ha='left', va='center')
            ledger.text(colx['Delta_Brier'], y, rendered[cid, 'Delta_Brier'], fontsize=7, ha='center', va='center')
    for yy in separators + [0.79]:
        fig.lines.append(Line2D([0.15 / W, 4.05 / W], [yy / H, yy / H], transform=fig.transFigure, color=ps.GRAY, linewidth=0.5))
    mean_geometry = []
    for name, key, yy in [('Held-out panel mean', 'original_T', 0.7), ('Scaffold-unseen mean', 'D_unseen_slice', 0.48)]:
        row = baseline['panel_mean_comparison'][key]
        labels.text(1, yy, name, fontsize=7, ha='right', va='center')
        plot.plot(row['point'], yy, 'D', color=ps.DARK_GRAY, markerfacecolor='white', markersize=6.5, markeredgewidth=0.5, markeredgecolor=ps.DARK_GRAY, zorder=2)
        plot.plot(row['ci95'], [yy, yy], color=ps.DARK_GRAY, linewidth=1.25, solid_capstyle='butt', zorder=3)
        plot.vlines(row['ci95'], yy - 1.5 / 72, yy + 1.5 / 72, color=ps.DARK_GRAY, linewidth=1.25, zorder=3)
        for x in [0.21, 0.69]:
            ledger.text(x, yy, '—', fontsize=7, ha='center', va='center')
        label = next((r['display'] for r in values if r['panel'] == 'A' and r['object'] == name and (r['field'] == 'point')))
        ledger.text(1.16, yy, label, fontsize=7, ha='center', va='center')
        mean_geometry.append({'object': name, 'point': row['point'], 'ci95': row['ci95'], 'horizontal_CI_span_pt': (row['ci95'][1] - row['ci95'][0]) / 0.00435 * 1.04 * 72, 'marker_size_pt': 6.5, 'CI_cap_height_pt': 3})
    for x, label, ha in [(4.2, 'Fit set', 'left'), (5.73, 'Six programs', 'center'), (7.25, 'Held-out set', 'right')]:
        text(fig, x, 5.53, label, fontsize=8, weight='bold', ha=ha)
    for x, label, ha in [(4.2, 'short proteins', 'left'), (5.73, 'X / S / L, ± unit', 'center'), (7.25, 'long proteins', 'right')]:
        text(fig, x, 5.34, label, fontsize=7, ha=ha)
    flow = axis(fig, 4.2, 5.4, 3.05, 0.26)
    flow.set_xlim(4.2, 7.25)
    flow.set_ylim(0, 1)
    flow.set_axis_off()
    for x0, x1 in [(4.79, 4.97), (6.42, 6.6)]:
        flow.annotate('', xy=(x1, 0.5), xytext=(x0, 0.5), arrowprops=dict(arrowstyle='-|>', color=ps.INK, lw=1, mutation_scale=7, shrinkA=0, shrinkB=0))
    text(fig, 4.2, 5.12, 'ESM-2 8M, block 3, selected unit')
    text(fig, 4.2, 4.94, 'Q5SI12 / 2CZL chain A / residues 161–182')
    protein_icons = []
    sequence_glyphs = []
    for kind, yy in [('HELIX', 4.7), ('STRAND', 4.25)]:
        site = example[kind + '_site_1based']
        masked = example[kind + '_mask_fragment_index']
        label = text(fig, 4.2, yy, '%s A%d' % (kind, site))
        label_width = label.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        protein_icons.append(secondary_icon(fig, 4.2 + label_width + 0.06, yy, kind))
        if kind == 'HELIX':
            local_sequence = sequence[:9]
            assert local_sequence == 'DLGAWWEER' and masked == 3 and (site == 164)
            sequence_glyphs.append(draw_sequence_window(fig, 4.2, yy - 0.3, local_sequence, 3, 0.16, font_pt=7.5))
        else:
            start, end = (174, 182)
            offset = start - example['fragment_start_1based']
            local_sequence = sequence[offset:offset + 9]
            local_mask = masked - offset
            assert local_sequence == 'LPLGAILAR' and local_mask == 4 and (site == 178)
            sequence_glyphs.append(draw_sequence_window(fig, 4.2, yy - 0.3, local_sequence, local_mask, 0.16, font_pt=7.5))
    key = axis(fig, 4.2, 3.83, 3.05, 0.13)
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.set_axis_off()
    key.plot(0.015, 0.5, 's', color=ps.BLUE, markersize=5.3, linestyle='None', clip_on=False)
    key.text(0.055, 0.5, 'Masked residue', fontsize=7, va='center')
    text(fig, 4.2, 3.6, 'C  Absolute prediction loss', fontsize=9, weight='bold', va='top')
    handles = [Line2D([], [], marker='o', linestyle='None', markerfacecolor=ps.ORANGE, markeredgecolor=ps.WHITE, markeredgewidth=0.5, markersize=6.5, label='With selected\nunit'), Line2D([], [], marker='o', linestyle='None', markerfacecolor=ps.WHITE, markeredgecolor=ps.BLUE, markeredgewidth=0.5, markersize=6.5, label='Without selected\nunit')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(4.2 / W, 3.4 / H), bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=7, handlelength=0.8, handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
    cp = axis(fig, 4.52, 2.51, 1.46, 0.72)
    cp.set_xlim(0.169, 0.201)
    cp.set_ylim(2.51, 3.23)
    cp.set_yticks([3.1, 2.88, 2.66])
    cp.set_yticklabels(['X', 'S', 'L'], fontsize=7)
    cp.set_xticks([0.17, 0.18, 0.19, 0.2])
    cp.set_xticklabels(['0.17', '0.18', '0.19', '0.20'], fontsize=7)
    cp.set_xlabel('Paired Brier loss', fontsize=8, labelpad=4)
    ps.clean_axes(cp)
    cp.spines['left'].set_visible(False)
    cp.tick_params(axis='y', length=0, pad=3)
    cl = axis(fig, 6.1, 2.51, 1.15, 0.72)
    cl.set_xlim(0, 1)
    cl.set_ylim(2.51, 3.23)
    cl.set_axis_off()
    for x, label in [(0.23, 'With\nunit'), (0.79, 'Without\nunit')]:
        text(fig, 6.1 + x * 1.15, 3.24, label, fontsize=7, ha='center')
    pairs = []
    for name, yy in [('X', 3.1), ('S', 2.88), ('L', 2.66)]:
        without = float(losses[name]['paired_Brier_mean'])
        with_unit = float(losses[name + 'j']['paired_Brier_mean'])
        cp.plot([without, with_unit], [yy, yy], color=ps.GRAY, linewidth=1)
        cp.plot(without, yy, 'o', markerfacecolor=ps.WHITE, markeredgecolor=ps.BLUE, markeredgewidth=0.5, markersize=6.5)
        cp.plot(with_unit, yy, 'o', color=ps.ORANGE, markeredgecolor=ps.WHITE, markeredgewidth=0.5, markersize=6.5)
        cl.text(0.23, yy, '%.3f' % with_unit, fontsize=7, ha='center', va='center')
        cl.text(0.79, yy, '%.3f' % without, fontsize=7, ha='center', va='center')
        pairs.append({'model': name, 'without_unit': without, 'with_unit': with_unit})
    text(fig, 4.2, 1.95, 'D  Paired loss reductions', fontsize=9, weight='bold', va='top')
    text(fig, 4.2, 1.72, '95% CI (thick); 98.75% CI (thin)')
    dp = axis(fig, 4.54, 0.43, 1.44, 0.94)
    dp.set_xlim(-0.0012, 0.031)
    dp.set_ylim(0.43, 1.37)
    dp.set_yticks([1.2, 0.98, 0.76, 0.54])
    dp.set_yticklabels(['gX', 'gS', 'gL', 'ψ =\ngX − gS'], fontsize=7)
    dp.set_xticks([0, 0.01, 0.02, 0.03])
    dp.set_xticklabels(['0', '0.01', '0.02', '0.03'], fontsize=7)
    dp.set_xlabel('Paired Brier reduction', fontsize=8, labelpad=4)
    ps.clean_axes(dp)
    dp.axvline(0, color='#999999', linewidth=0.5, linestyle='--', zorder=0)
    dp.spines['left'].set_visible(False)
    dp.tick_params(axis='y', length=0, pad=3)
    dl = axis(fig, 6.1, 0.43, 1.15, 0.94)
    dl.set_xlim(0, 1)
    dl.set_ylim(0.43, 1.37)
    dl.set_axis_off()
    for x, label in [(0.2, 'Gain'), (0.83, 'Holm p')]:
        text(fig, 6.1 + x * 1.15, 1.48, label, fontsize=7, weight='bold', ha='center')
    for endpoint, yy in [('gX', 1.2), ('gS', 0.98), ('gL', 0.76), ('psi', 0.54)]:
        row = endpoints[endpoint]
        color = ps.ORANGE
        for lower, upper, width in [('CI98_75_lower', 'CI98_75_upper', 0.65), ('CI95_lower', 'CI95_upper', 1.25)]:
            dp.plot([float(row[lower]), float(row[upper])], [yy, yy], color=color, linewidth=width, solid_capstyle='butt')
        dp.plot(float(row['mean']), yy, marker='o' if endpoint != 'psi' else 'D', color=color, markersize=6.5, markeredgecolor=ps.WHITE, markeredgewidth=0.5)
        gain = next((r['display'] for r in values if r['panel'] == 'D' and r['object'] == endpoint and (r['field'] == 'mean')))
        p = next((r['display'] for r in values if r['panel'] == 'D' and r['object'] == endpoint and (r['field'] == 'Holm_p')))
        dl.text(0.2, yy, gain, fontsize=7, ha='center', va='center')
        dl.text(1, yy, p, fontsize=7, ha='right', va='center')
    geometry = {'shared_radius1_glyph': radius_glyph, 'shared_sequence_windows': sequence_glyphs, 'A164_view_original_range_1based': [161, 169], 'A178_view_original_range_1based': [174, 182], 'scientific_input_original_range_1based': [161, 182], 'panel_bounds_in': panels, 'outer_panel_margin_in': 0.15, 'right_column_panel_gaps_in': [0.15, 0.15], 'A_row_spacing_in': 0.22, 'C_row_spacing_in': 0.22, 'D_row_spacing_in': 0.22, 'A_mean_row_spacing_in': 0.22, 'A_group_separator_clearance_in': 0.08, 'A_row_positions_in': row_positions, 'A_group_header_centers_in': headers, 'A_separator_positions_in': separators, 'A_axes_bounds_in': [1.62, 0.4, 1.04, 4.38], 'C_axes_bounds_in': [4.52, 2.51, 1.46, 0.72], 'D_axes_bounds_in': [4.54, 0.43, 1.44, 0.94], 'group_representative_SMILES_placements': placements, 'protein_class_icons': protein_icons, 'amide_native_size_in': [molecule['width_pt'] / 72, molecule['height_pt'] / 72], 'amide_position_in': [0.15, 4.95], 'amide_CPK_geometry': molecule_path.as_posix(), 'mean_CI_geometry': mean_geometry, 'whole_asset_scaling': False, 'AP_microbar_height_in': 0.08, 'AP_microbar_range': [0, 1], 'AP_microbar_color': '#DDDDDD', 'CPK_palette': PALETTE, 'family_colors': {k: v[0] for k, v in FAMILIES.items()}, 'family_markers': {k: v[1] for k, v in FAMILIES.items()}, 'A_legend_family_color_text': None, 'A_legend_dark_gray_lookup_only_removed': True}
    data = copy.deepcopy(baseline)
    data.update({'figure': 3, 'version': version, 'preceding_scientific_baseline': SOURCE.as_posix(), 'physical_layout': geometry, 'group_query_placements': placements, 'protein_class_icons': protein_icons, 'categorical_display_row_positions_in': row_positions, 'display_values': values, 'C_plot_pairs': pairs, 'all72_source_display_records_exactly_reused': True, 'new_fits_forwards_bootstraps_p_tests': 0})
    meta = {'panels': {name: rect(*bounds) for name, bounds in panels.items()}, 'physical_layout': geometry, 'display_value_count': 72, 'legend_entries': 5, 'baseline_source': SOURCE.as_posix(), 'scientific_display_counts': {'concept_rows': 12, 'panel_means': 2, 'absolute_losses': 6, 'protein_endpoints': 4}, 'absolute_loss_CI': False, 'new_scientific_estimation': False}
    archive, meta = save_fixed(fig, 3, version, data, meta)
    meta['caption_status'] = 'Frozen figure source'
    for directory in [archive]:
        write(directory / 'figure_3_metadata.json', meta)
    for directory in [archive]:
        write(directory / 'figure_3_geometry.json', geometry)
        with (directory / 'figure_3_value_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print('Figure 3:', meta['PNG_pixel_size'], meta['width_in'], meta['height_in'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='portable')
    main(parser.parse_args().version)
