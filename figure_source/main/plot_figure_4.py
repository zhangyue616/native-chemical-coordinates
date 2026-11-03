"""Portable rendering from frozen values and geometry."""
import argparse
import copy
import math
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from fig4_helpers import ROOT, BASE, FROZEN, ps, read, save
R8 = ROOT / 'source_data/inputs'
SOURCE = R8 / 'figure_4_baseline.json'
SELECTION = FROZEN / 'selected_saved_atom_values_20261102.json'
W, H = (8.73, 4.65)
METHODS = ['D_mean', 'M1', 'M2', 'M1_delta_perm', 'M2_delta_perm']
LABELS = {'D_mean': 'Fit mean', 'M1': 'Concept mean', 'M2': 'Local chemistry', 'M1_delta_perm': 'Concept shuffled', 'M2_delta_perm': 'Local shuffled'}
SYMBOLS = {'D_mean': dict(color=ps.GRAY, marker='D', filled=False), 'M1': dict(color=ps.BLUE, marker='o', filled=True), 'M2': dict(color=ps.TEAL, marker='s', filled=True), 'M1_delta_perm': dict(color=ps.BLUE, marker='o', filled=False), 'M2_delta_perm': dict(color=ps.TEAL, marker='s', filled=False)}
TARGETS = [(293, 26, 115, 'Aliphatic OH'), (454, 24, 124, 'Carbonyl')]
STRATA = [('all_T', 'All'), ('any_target_positive_molecule', 'Concept-positive'), ('all_target_negative_molecule', 'All-negative')]
C_LEFT, C_RIGHT, C_GAP = (3.65, 8.63, 0.08)
C_SCALE, HALO_RADIUS_IN, CBAR_WIDTH_IN = (1.4, 0.12, 1.2)
MARKER_PT, B_MARKER_PT, SHUFFLED_UP_IN = (6.5, 5.0, 0.0)

def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]

def text(fig, x, y, s, **kw):
    opts = dict(fontsize=7.5, va='center')
    opts.update(kw)
    return fig.text(x / W, y / H, s, **opts)

def clean(ax):
    ps.clean_axes(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=4, labelsize=7.5)
    ax.tick_params(axis='x', labelsize=7.5, pad=2)

def interval(ax, values, y, condition, factor=1, marker_pt=MARKER_PT):
    point, lo, hi = [float(v) * factor for v in values]
    s = SYMBOLS[condition]
    ax.plot([lo, hi], [y, y], color=s['color'], lw=1.25, solid_capstyle='butt', zorder=2)
    line, = ax.plot(point, y, ls='none', marker=s['marker'], ms=marker_pt, mfc=s['color'] if s['filled'] else ps.WHITE, mec=ps.WHITE if s['filled'] else s['color'], mew=0.5, zorder=3 if s['filled'] else 4)
    return line

def pieces(fig, x, y, runs):
    renderer = fig.canvas.get_renderer()
    widths = [renderer.get_text_width_height_descent(s, FontProperties(family='Arial', size=size), False)[0] / fig.dpi for s, size, dy in runs]
    cursor = -sum(widths) / 2
    for (s, size, dy), width in zip(runs, widths):
        text(fig, x + cursor, y + dy / 72, s, fontsize=size)
        cursor += width

def exponent_label(fig, ax, label, labelpad=2, coords=None):
    t = ax.set_xlabel(label + ' × 10', fontsize=8, labelpad=labelpad)
    if coords is not None:
        ax.xaxis.set_label_coords(*coords)
    fig.canvas.draw()
    box = t.get_window_extent(fig.canvas.get_renderer())
    text(fig, box.x1 / fig.dpi + 0.005, box.y0 / fig.dpi + 0.045, '−3', fontsize=7.5, ha='left', va='bottom')

def cut_marks(fig, low, high):
    a, b = (low.get_position().x1, high.get_position().x0)
    for y in [low.get_position().y0, low.get_position().y1 - 0.025 / H]:
        for f in [0.25, 0.75]:
            x = a + f * (b - a)
            fig.lines.append(Line2D([x - 0.014 / W, x + 0.014 / W], [y - 0.024 / H, y + 0.024 / H], transform=fig.transFigure, color=ps.INK, lw=0.9))

def historical(ax, values, y):
    p, lo, hi = [float(v) * 1000 for v in values]
    ax.plot([lo, hi], [y, y], color=ps.ORANGE, lw=1.25, solid_capstyle='butt')
    ax.plot(p, y, 'o', color=ps.ORANGE, ms=MARKER_PT, mew=0.5, mec=ps.WHITE)

def enlarged_geometry(molecule):
    """Scale the frozen physical atom/bond coordinates, never a final image."""
    from chemical_asset_style import _align_atom_text
    from matplotlib.textpath import TextPath, TextToPath
    p = FROZEN / ('molecule_%s_raw_geometry.json' % molecule['molecule_id'])
    g = copy.deepcopy(read(p))
    for a in g['atoms']:
        a['x'] *= C_SCALE
        a['y'] *= C_SCALE
    for path in g['paths']:
        path['vertices'] = [[x * C_SCALE, y * C_SCALE] for x, y in path['vertices']]
    for t in g['texts']:
        t['x'] *= C_SCALE
        t['y'] *= C_SCALE
    _align_atom_text(g)
    radius_pt = HALO_RADIUS_IN * 72
    boxes = [[a['x'] - radius_pt, a['y'] - radius_pt, a['x'] + radius_pt, a['y'] + radius_pt] for a in g['atoms']]
    for t in g['texts']:
        prop = FontProperties(family='Arial', size=7)
        box = TextPath((0, 0), t['text'], size=7, prop=prop).get_extents()
        advance = TextToPath().get_text_width_height_descent(t['text'], prop, False)[0]
        x = t['x'] - {'middle': advance / 2, 'end': advance}.get(t.get('anchor'), 0)
        boxes.append([x + box.x0, t['y'] - box.y1, x + box.x1, t['y'] - box.y0])
    bounds = [min((b[0] for b in boxes)), min((b[1] for b in boxes)), max((b[2] for b in boxes)), max((b[3] for b in boxes))]
    pad_pt = 0.5
    padded = [bounds[0] - pad_pt, bounds[1] - pad_pt, bounds[2] + pad_pt, bounds[3] + pad_pt]
    lengths = [math.hypot(g['atoms'][b['begin']]['x'] - g['atoms'][b['end']]['x'], g['atoms'][b['begin']]['y'] - g['atoms'][b['end']]['y']) for b in g['bonds']]
    assert max((abs(v - 11.2) for v in lengths)) < 1e-06
    return (g, p, bounds, padded, lengths)

def draw_activation(fig, selected):
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
    from matplotlib.cm import ScalarMappable
    cmap = LinearSegmentedColormap.from_list('white_orange', ['#FFFFFF', '#EE7733'], N=256)
    assert to_hex(cmap(1.0)).upper() == '#EE7733'
    prepared = [enlarged_geometry(m) for m in selected['molecules']]
    widths = [(p[3][2] - p[3][0]) / 72 for p in prepared]
    cell_width = max(widths)
    needed_width = 3 * cell_width + 2 * C_GAP
    assert needed_width <= C_RIGHT - C_LEFT + 1e-09
    centers = [C_LEFT + cell_width / 2 + i * (cell_width + C_GAP) for i in range(3)]
    for x, label in zip(centers, ['Observed', 'M1', 'M2']):
        text(fig, x, 4.43, label, fontsize=8, ha='center', weight='bold')
    records = []
    for ri, (m, prepared_row) in enumerate(zip(selected['molecules'], prepared)):
        g, p, ink, padded, lengths = prepared_row
        x0, y0, x1, y1 = padded
        width, height = ((x1 - x0) / 72, (y1 - y0) / 72)
        arrays = [m[k] for k in ['observed', 'M1', 'M2']]
        assert all((len(a) == m['n_atoms'] for a in arrays))
        assert m['graph_atom_ids'] == list(range(m['n_atoms']))
        vmin, vmax = (min((min(a) for a in arrays)), max((max(a) for a in arrays)))
        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
        title_y, center_y = [(4.16, 3.55), (2.76, 2.15)][ri]
        row_title = ['Aliphatic OH unit', 'Carbonyl unit'][ri]
        text(fig, C_LEFT, title_y, row_title, fontsize=8, weight='bold')
        placements = []
        for x, label, values in zip(centers, ['Observed', 'M1', 'M2'], arrays):
            left, bottom = (x - width / 2, center_y - height / 2)
            ax = fig.add_axes(rect(left, bottom, width, height))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect('equal')
            ax.axis('off')
            ax._chemical_asset_audit = {'native_size_in': [width, height], 'scaling_applied': False, 'font_pt': 7}
            for a in sorted(g['atoms'], key=lambda a: values[int(a['index'])]):
                value = values[int(a['index'])]
                ax.add_patch(Circle((a['x'], a['y']), HALO_RADIUS_IN * 72, facecolor=cmap(norm(value)), edgecolor='none', alpha=1, zorder=1))
            for path in g['paths']:
                ax.add_patch(PathPatch(MplPath(path['vertices'], path['codes']), facecolor='none', edgecolor='#20252A', linewidth=0.6, capstyle='round', joinstyle='round', zorder=2))
            for t in g['texts']:
                ax.text(t['x'], t['y'], t['text'], fontsize=7, family='Arial', va='baseline', color='#20252A', ha={'start': 'left', 'middle': 'center', 'end': 'right'}.get(t.get('anchor'), 'left'), zorder=3)
            placements.append({'method': label, 'bounds_in': [left, bottom, width, height], 'ink_and_halo_bounds_in': [left + (ink[0] - x0) / 72, bottom + (y1 - ink[3]) / 72, (ink[2] - ink[0]) / 72, (ink[3] - ink[1]) / 72]})
        cbounds = [C_RIGHT - CBAR_WIDTH_IN, title_y - 0.035, CBAR_WIDTH_IN, 0.07]
        cbax = fig.add_axes(rect(*cbounds))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbax, orientation='horizontal')
        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels(['min ' + format(vmin, '.3g'), 'max ' + format(vmax, '.3g')])
        cb.ax.tick_params(labelsize=7.5, length=2, pad=2)
        ticks = cb.ax.get_xticklabels()
        ticks[0].set_ha('left')
        ticks[1].set_ha('right')
        cb.outline.set_linewidth(0.5)
        if cb.solids is not None:
            cb.solids.set_rasterized(False)
        records.append({'molecule': copy.deepcopy(m), 'raw_geometry_source': str(p), 'placements': placements, 'row_title': row_title, 'title_y_in': title_y, 'colormap': 'linear white #FFFFFF to orange #EE7733', 'max_RGBA': list(cmap(norm(vmax))), 'halo_alpha': 1.0, 'halo_paint_order': 'ascending original value, saturated maximum last', 'shared_color_range': [vmin, vmax], 'normalization': 'one common original linear range over all three saved arrays for this molecule; no clipping', 'negative_values_retained': True, 'atom_center_bond_length_pt': 11.2, 'measured_center_bond_length_range_pt': [min(lengths), max(lengths)], 'R8_atom_center_bond_length_pt': 8.0, 'physical_coordinate_multiplier': C_SCALE, 'whole_final_asset_scaled': False, 'atom_font_pt': 7.0, 'bond_linewidth_pt': 0.6, 'activation_disk_radius_in': HALO_RADIUS_IN, 'activation_disk_radius_pt': HALO_RADIUS_IN * 72, 'ink_and_halo_size_in': [(ink[2] - ink[0]) / 72, (ink[3] - ink[1]) / 72], 'colorbar_bounds_in': cbounds, 'colorbar_ticks': [vmin, vmax], 'colorbar_tick_labels': ['min ' + format(vmin, '.3g'), 'max ' + format(vmax, '.3g')], 'colorbar_midpoint_shown': False, 'CPK_colors_used': False})
    return {'selection_source': str(SELECTION), 'source_metadata': {k: v for k, v in selected.items() if k != 'molecules'}, 'molecules': records, 'C_cell_width_in': cell_width, 'C_required_width_in': needed_width, 'C_horizontal_cell_gap_in': C_GAP, 'predictions_recomputed': False, 'observations_or_predictions_interpolated': False}

def marker_overlap_records(fig, ax, rows):
    """Check the actual configured marker paths at their final physical positions."""
    groups = {}
    for r in rows:
        groups.setdefault(r['label'], []).append(r)
    report = []
    for label, group in groups.items():
        glyphs = []
        for r in group:
            c = r['condition']
            x, y = ax.transData.transform((r['plot_values_original_units'][0] * 1000, r['display_marker_y'])) / fig.dpi
            marker = MarkerStyle(SYMBOLS[c]['marker'])
            path = marker.get_path().transformed(marker.get_transform())
            path = path.transformed(Affine2D().scale(B_MARKER_PT / 72).translate(x, y))
            glyphs.append((c, path, [float(x), float(y)]))
        intersections = []
        for i, (ca, pa, xy_a) in enumerate(glyphs):
            for cb, pb, xy_b in glyphs[i + 1:]:
                if pa.intersects_path(pb, filled=True):
                    intersections.append({'methods': [ca, cb], 'center_distance_in': math.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1]), 'contains_shuffled': ca.endswith('_perm') or cb.endswith('_perm')})
        report.append({'row': label, 'intersections': intersections})
    return report

def build(version):
    old = read(SOURCE)
    selected = read(SELECTION)
    fig = ps.new_figure(W, H)
    rowsA, rowsB = ([], [])
    text(fig, 0.1, 4.43, 'A  Activation R²', fontsize=9, weight='bold')
    abounds = [1.22, 3.88, 2.15, 0.42]
    axa = fig.add_axes(rect(*abounds))
    clean(axa)
    for i, (coordinate, concept, endpoint, name) in enumerate(TARGETS):
        y, means = (1 - i, [])
        for c in METHODS[:3]:
            r = copy.deepcopy(next((r for r in old['activation_rows'] if r['condition'] == c and int(r['source_row']['coordinate']) == coordinate)))
            vals = r['plot_values']
            means.append(vals[0])
            interval(axa, vals, y, c)
            r.update(display_row_y=y, label=name)
            rowsA.append(r)
        axa.plot(means, [y] * 3, color='#CCCCCC', lw=0.75, zorder=1)
    axa.set_ylim(-0.5, 1.5)
    axa.set_xlim(-0.035, 0.9)
    axa.set_yticks([1, 0])
    axa.set_yticklabels(['Aliphatic OH', 'Carbonyl'])
    axa.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axa.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
    pieces(fig, 2.295, 3.62, [('R', 8, 0), ('2', 7.5, 3.1)])
    text(fig, 0.1, 3.28, 'B  Output MAE (log scale)', fontsize=9, weight='bold')
    bbounds = [1.86, 1.98, 1.51, 1.07]
    axb = fig.add_axes(rect(*bbounds))
    clean(axb)
    axb.set_xscale('log')
    axb.set_xlim(0.2, 200)
    axb.set_ylim(-0.5, 5.5)
    axb.xaxis.set_major_locator(FixedLocator([0.2, 1, 10, 100]))
    axb.xaxis.set_major_formatter(FixedFormatter(['0.2', '1', '10', '100']))
    axb.xaxis.set_minor_locator(NullLocator())
    row_labels = []
    dy = SHUFFLED_UP_IN / (bbounds[3] / 6)
    for ti, (coordinate, concept, endpoint, name) in enumerate(TARGETS):
        for si, (stratum, label) in enumerate(STRATA):
            y, means = (5 - (ti * 3 + si), [])
            for c in METHODS:
                r = copy.deepcopy(next((r for r in old['output_rows'] if r['condition'] == c and int(r['source_row']['coordinate']) == coordinate and (r['source_row']['stratum'] == stratum))))
                vals = r['plot_values_original_units']
                y_display = y + dy if c.endswith('_perm') else y
                interval(axb, vals, y_display, c, 1000, marker_pt=B_MARKER_PT)
                if c in METHODS[:3]:
                    means.append(float(vals[0]) * 1000)
                r.update(display_row_y=y, display_marker_y=y_display, display_CI_y=y_display, display_upward_offset_in=SHUFFLED_UP_IN if c.endswith('_perm') else 0, label=name + ' / ' + label)
                rowsB.append(r)
            axb.plot(means, [y] * 3, color='#CCCCCC', lw=0.75, zorder=1)
            n = int(r['source_row']['n_molecules'])
            row_labels.append('%s (n = %s)' % (label, format(n, ',')))
    axb.set_yticks([5, 4, 3, 2, 1, 0])
    axb.set_yticklabels(row_labels)
    text(fig, 0.1, 3.1, 'Aliphatic OH', weight='bold')
    text(fig, 0.1, 2.55, 'Carbonyl', weight='bold')
    exponent_label(fig, axb, 'MAE', labelpad=3)
    handles = [Line2D([], [], ls='none', marker=SYMBOLS[c]['marker'], ms=MARKER_PT, mfc=SYMBOLS[c]['color'] if SYMBOLS[c]['filled'] else ps.WHITE, mec=ps.WHITE if SYMBOLS[c]['filled'] else SYMBOLS[c]['color'], mew=0.5, label=LABELS[c]) for c in METHODS]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(2.1 / W, 1.41 / H), bbox_transform=fig.transFigure, ncol=3, frameon=False, fontsize=7.5, handlelength=0.9, handletextpad=0.35, columnspacing=0.8, labelspacing=0.4)
    text(fig, C_LEFT, 4.57, 'C  Atomwise activation', fontsize=9, weight='bold')
    cdata = draw_activation(fig, selected)
    text(fig, 0.1, 1.14, 'D  Intervention contrasts', fontsize=9, weight='bold')
    dbounds = [[2.13, 0.42, 1.2, 0.48], [3.43, 0.42, 1.4, 0.48], [6.38, 0.42, 2.05, 0.48]]
    low, high, pair = [fig.add_axes(rect(*b)) for b in dbounds]
    full_title = text(fig, 2.73, 0.99, 'Full clamp', fontsize=8, weight='bold', ha='center')
    text(fig, 7.405, 0.99, 'Paired dose', fontsize=8, weight='bold', ha='center')
    for ax in [low, high, pair]:
        clean(ax)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([])
        ax.tick_params(axis='x', pad=0)
    low.set(xlim=(-0.2, 5), xticks=[0, 2, 4])
    high.set(xlim=(120, 170), xticks=[130, 160])
    pair.set(xlim=(-1.4, 1.05), xticks=[-1, 0, 1])
    pair.axvline(0, color='#999999', lw=0.5, ls='--', zorder=0)
    low.axvline(0, color='#999999', lw=0.5, ls='--', zorder=0)
    cut_marks(fig, low, high)
    for r in old['historical_contrast_rows']:
        y = r['display_row_y']
        historical(low, r['full_clamp_original_precision'], y)
        historical(high, r['full_clamp_original_precision'], y)
        historical(pair, r['paired_dose_original_precision'], y)
        text(fig, 1.98, 0.42 + (y + 0.5) / 3 * 0.48, r['name'], ha='right')
    exponent_label(fig, high, 'Contrast', labelpad=1, coords=(1.1, -0.5))
    fig.canvas.draw()
    overlaps = marker_overlap_records(fig, axb, rowsB)
    renderer = fig.canvas.get_renderer()
    full_title_right = full_title.get_window_extent(renderer).x1 / fig.dpi
    break_left = dbounds[0][0] + dbounds[0][2] + 0.25 * 0.1 - 0.014
    assert full_title_right < break_left
    geometry = {'canvas_in': [W, H], 'R8_canvas_in': [7, 5.8], 'minimum_width_for_C_at_frozen_left_boundary_in': C_LEFT + cdata['C_required_width_in'] + 0.1, 'layout_slots_in': {'A': [0.1, 3.55, 3.27, 1.0], 'B': [0.1, 1.6, 3.27, 1.8], 'C': [C_LEFT, 1.6, C_RIGHT - C_LEFT, 3.05], 'D': [0.1, 0.03, W - 0.2, 1.2]}, 'axes_bounds_in': {'A': abounds, 'B': bbounds, 'D': dbounds}, 'A_method_rows': 2, 'A_dumbbell_linewidth_pt': 0.75, 'B_method_rows': 6, 'B_row_spacing_in': bbounds[3] / 6, 'B_log_x_limits': [0.2, 200], 'B_marker_size_pt': B_MARKER_PT, 'B_shuffled_scheme': 'Frozen figure source', 'B_first_scheme_evidence': 'r9_20261102_v01: actual 6.5pt markers and CI up 0.06in; residual intersections remained', 'B_shuffled_upward_offset_in': SHUFFLED_UP_IN, 'B_additional_jitter': False, 'B_n_in_row_labels': True, 'B_actual_marker_intersections': overlaps, 'B_overlap_eliminated': not any((r['intersections'] for r in overlaps)), 'C_reserved_bounds_in': [C_LEFT, 1.6, C_RIGHT - C_LEFT, 3.05], 'C_saved_array_rendered': True, 'C_molecule_coordinate_multiplier': C_SCALE, 'C_native_bond_length_pt': 11.2, 'C_halo_radius_in': HALO_RADIUS_IN, 'C_colorbar_width_in': CBAR_WIDTH_IN, 'C_molecule_rows': [{k: r[k] for k in ['row_title', 'placements', 'ink_and_halo_size_in', 'colorbar_bounds_in', 'shared_color_range', 'measured_center_bond_length_range_pt']} for r in cdata['molecules']], 'D_full_clamp_segments_display': [[-0.2, 5], [120, 170]], 'D_full_clamp_gap_display': [5, 120], 'D_paired_dose_limits': [-1.4, 1.05], 'D_row_spacing_in': 0.16, 'D_full_clamp_title_right_in': full_title_right, 'D_break_left_ink_in': break_left, 'D_title_to_break_horizontal_clearance_in': break_left - full_title_right, 'method_symbols': SYMBOLS, 'all_frozen_CI_values_unchanged': True, 'whole_figure_scaled': False, 'old_AB_fragment_glyphs_retained': False, 'visible_C_internal_IDs': False, 'C_IDs_and_coordinates_in_caption': True}
    data = copy.deepcopy(old)
    data.update(version=version, R8_frozen_source=str(SOURCE), new_statistics=False, activation_rows=rowsA, output_rows=rowsB, C_saved_atom_values=cdata, physical_layout=geometry)
    data['caption_facts'].update({'B_display': 'Frozen figure source', 'C_display': 'rows titled Aliphatic OH unit and Carbonyl unit; CID and j293/j454 removed from figure and retained in caption; row-specific white to #EE7733 maps with true max saturated and negative minima retained; horizontal 1.2in right-side min/max colorbars'})
    target, meta = save(fig, version, data, geometry, Path(__file__))
    print(str(target))
    print(meta['PNG_pixel_size'])
    for row in overlaps:
        print(row['row'], [r['methods'] for r in row['intersections']])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='portable')
    args = parser.parse_args()
    build(args.version)
