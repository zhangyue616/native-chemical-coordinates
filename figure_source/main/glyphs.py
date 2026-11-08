"""Portable rendering from frozen values and geometry."""
"""Shared scientific glyphs for the 2026-11-02 figure set.

All placement arguments are physical inches, with (x, y) at the lower left.
No statistical values are represented by these schematic state markers.
"""
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
J1, J2, REFERENCE = ('#33BBEE', '#EE3377', '#999999')
BLUE, ORANGE = ('#0077BB', '#EE7733')

def _line(fig, p, q, color='black', lw=0.5, **kw):
    fw, fh = fig.get_size_inches()
    line = Line2D([p[0] / fw, q[0] / fw], [p[1] / fh, q[1] / fh], transform=fig.transFigure, color=color, linewidth=lw, **kw)
    line.set_figure(fig)
    fig.lines.append(line)
    return line

def _bounds(x, y, w, h, kind, **metadata):
    return dict(kind=kind, bounds_in=[x, y, x + w, y + h], placement_in=[x, y, w, h], **metadata)

def draw_state_matrix(fig, x, y, size_in, metric=None, grid=True):
    """Native left/Rotated right; Trained top/Initialization bottom.

    metric=None draws J1, J2 and the reference. metric='J1' or 'J2' keeps
    its colored arrow and the gray reference, without inventing data points.
    Markers are filled o/D (trained), open o/D (initialization).
    """
    fw, fh = fig.get_size_inches()
    s = float(size_in)
    lw = 0.5 if s < 0.4 else 0.5
    if grid:
        for frac in (0, 0.5, 1):
            _line(fig, (x + frac * s, y), (x + frac * s, y + s), '#DDDDDD', lw)
            _line(fig, (x, y + frac * s), (x + s, y + frac * s), '#DDDDDD', lw)
    centers = {'trained_native': (x + 0.25 * s, y + 0.75 * s), 'trained_rotated': (x + 0.75 * s, y + 0.75 * s), 'init_native': (x + 0.25 * s, y + 0.25 * s), 'init_rotated': (x + 0.75 * s, y + 0.25 * s)}
    ms = min(6.5, s * 72 * 0.14)
    for name, (cx, cy) in centers.items():
        native = name.endswith('native')
        marker = 'o' if native else 'D'
        point = Line2D([cx / fw], [cy / fh], transform=fig.transFigure, marker=marker, markersize=ms, linestyle='None', markerfacecolor='black' if name.startswith('trained') else 'white', markeredgecolor='black', markeredgewidth=0.5)
        point.set_figure(fig)
        fig.lines.append(point)
    arrows = []
    for role, a, b, color in [('J1', 'init_native', 'trained_native', J1), ('J2', 'trained_rotated', 'trained_native', J2), ('reference', 'init_rotated', 'init_native', REFERENCE)]:
        if role != 'reference' and metric is not None and (metric != role):
            continue
        p, q = (centers[a], centers[b])
        arrow = FancyArrowPatch((p[0] / fw, p[1] / fh), (q[0] / fw, q[1] / fh), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=max(1.8, min(6, s * 13)), linewidth=0.75 if s >= 0.5 else 0.5, color=color, shrinkA=ms * 0.7, shrinkB=ms * 0.7)
        arrow.set_figure(fig)
        fig.patches.append(arrow)
        arrows.append(dict(role=role, start_in=list(p), end_in=list(q), color=color))
    return _bounds(x, y, s, s, 'state_matrix', column_order=['Native', 'Rotated'], row_order_top_to_bottom=['Trained', 'Initialization'], marker_order=['filled o', 'filled D', 'open o', 'open D'], centers_in=centers, arrows=arrows, metric=metric, schematic_not_statistical_values=True)

def draw_radius1(fig, x, y, size_in, circle_diameter_in=None):
    """Center plus three neighbors; .14in circles in the full-size glyph.

    The .2in title thumbnail has geometry only and scales proportionately
    to the shared .46in full glyph with .14in circles.
    """
    fw, fh = fig.get_size_inches()
    s = float(size_in)
    d = float(circle_diameter_in) if circle_diameter_in is not None else 0.14 * s / 0.46
    r = d / 2
    centers = [(x + 0.5 * s, y + 0.48 * s), (x + r, y + r), (x + s - r, y + r), (x + 0.5 * s, y + s - r)]
    for p in centers[1:]:
        a = centers[0]
        dx, dy = (p[0] - a[0], p[1] - a[1])
        length = math.hypot(dx, dy)
        _line(fig, (a[0] + dx * r / length, a[1] + dy * r / length), (p[0] - dx * r / length, p[1] - dy * r / length), lw=0.75)
    for i, (cx, cy) in enumerate(centers):
        from matplotlib.patches import Ellipse
        fig.patches.append(Ellipse((cx / fw, cy / fh), d / fw, d / fh, transform=fig.transFigure, facecolor='white', edgecolor='black', linewidth=0.75))
    return _bounds(x, y, s, s, 'radius1', center_count=1, neighbor_count=3, circle_diameter_in=d, bond_linewidth_pt=0.75, centers_in=centers)

def draw_token_grid(fig, x, y, size_in, rows=4, cols=4, orange_col=1):
    """Atoms are rows, coordinates are columns; exactly one orange column."""
    fw, fh = fig.get_size_inches()
    s = float(size_in)
    cw = s / cols
    ch = s / rows
    for row in range(rows):
        for col in range(cols):
            fig.patches.append(Rectangle(((x + col * cw) / fw, (y + row * ch) / fh), cw / fw, ch / fh, transform=fig.transFigure, facecolor=ORANGE if col == orange_col else 'white', edgecolor='black', linewidth=0.5))
    return _bounds(x, y, s, s, 'atoms_coordinates_grid', rows=rows, cols=cols, row_semantics='atoms', column_semantics='coordinates', orange_column_zero_based=orange_col, orange_columns=1)

def draw_sequence_window(fig, x, y, sequence, mask_index, cell_in, font_pt=7.5, mask_color=BLUE, mask_text='[MASK]', cell_height_in=None):
    """Fixed-width residue cells; the masked original target is not input.

    The target cell is a blue block without a residue letter. Its original
    residue/position are returned as metadata for the caption and source audit.
    """
    if font_pt < 7:
        raise ValueError('Residue font must remain at least 7pt')
    if not 0 <= mask_index < len(sequence):
        raise ValueError('mask_index is zero-based within sequence')
    fw, fh = fig.get_size_inches()
    w = len(sequence) * cell_in
    h = cell_in if cell_height_in is None else float(cell_height_in)
    if h <= 0:
        raise ValueError('cell_height_in must be positive')
    for i, residue in enumerate(sequence):
        masked = i == mask_index
        fig.patches.append(Rectangle(((x + i * cell_in) / fw, y / fh), cell_in / fw, h / fh, transform=fig.transFigure, facecolor=mask_color if masked else 'white', edgecolor=mask_color if masked else '#DDDDDD', linewidth=0.5))
        if not masked:
            fig.text((x + (i + 0.5) * cell_in) / fw, (y + h / 2) / fh, residue, fontsize=font_pt, family='Arial', ha='center', va='center', color='black')
    return _bounds(x, y, w, h, 'masked_sequence_window', sequence_original=sequence, mask_index_zero_based=mask_index, original_target_residue=sequence[mask_index], model_input_residues=[mask_text if i == mask_index else r for i, r in enumerate(sequence)], original_target_is_not_supplied_as_input=True, mask_color=mask_color, cell_in=cell_in, cell_width_in=cell_in, cell_height_in=h, font_pt=font_pt, alignment='equal-width Arial residue cells')
