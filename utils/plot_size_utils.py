import matplotlib
import matplotlib.pyplot as plt
import math


#for all matplotlib plots, set the default font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.major.width'] = 0.72
plt.rcParams['ytick.major.width'] = 0.72
plt.rcParams['xtick.minor.width'] = 0.72
plt.rcParams['ytick.minor.width'] = 0.72
#set font size to 8
dfontsize = 8
plt.rcParams.update({'font.size': dfontsize})
default_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1, 'wspace': 0.01, 'hspace': 0.01}
summary_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.3, 'wspace': 0.01, 'hspace': 0.01}
dPanW = 1.2
dPanH = 0.8
dPad = round(0.2/(8*(1/72)), 2)
dh_pad = round(0.09/(8*(1/72)), 2)

def detect_subplot_grid_shape(fig):
    """
    Attempt to detect the number of rows and columns of 'main' subplots
    in a figure by examining their positions in normalized figure coordinates.

    This assumes a rectangular grid of subplots laid out by something
    like plt.subplots(nrows, ncols) (i.e. uniform sized Axes).

    Returns
    -------
    (nrows, ncols) : tuple of int
    """
    # Get all Axes
    axes = fig.get_axes()

    # Filter to only "real" subplot Axes. We skip things like colorbars or inset_axes
    # A simple heuristic: check if it's an instance of SubplotBase.
    # Alternatively, you could skip any axes that appear to be legends or colorbars.
    real_subplots = []
    for ax in axes:
        # We also skip invisible or twin axes (to avoid duplicates).
        if (ax.get_visible() and
                isinstance(ax, matplotlib.axes.SubplotBase)):
            real_subplots.append(ax)

    if not real_subplots:
        return 0, 0  # No valid subplots found

    # Extract bounding boxes in normalized figure coords
    # We'll look at the center or the lower-left corner.
    # For consistent subplots, either approach works if they're arranged in a grid.
    x_positions = []
    y_positions = []
    for ax in real_subplots:
        bbox = ax.get_position()  # returns Bbox(x0, y0, x1, y1) in [0..1]
        # Let's use the center to avoid minor floating offsets on edges
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        x_positions.append(x_center)
        y_positions.append(y_center)

    # Group unique centers for x and y, which correspond to columns and rows respectively
    # We'll define a small tolerance to group subplots in the same row/column
    # if their centers differ by less than some tiny threshold.
    def unique_positions(vals, tol=1e-3):
        # Sort them
        vals_sorted = sorted(vals)
        unique_vals = []
        current_group = None
        for v in vals_sorted:
            if current_group is None:
                current_group = v
                unique_vals.append(v)
            else:
                # if difference is large enough, we treat as a new group
                if abs(v - current_group) > tol:
                    unique_vals.append(v)
                    current_group = v
        return unique_vals

    unique_x = unique_positions(x_positions, tol=1e-3)
    unique_y = unique_positions(y_positions, tol=1e-3)

    ncols = len(unique_x)
    nrows = len(unique_y)

    return (nrows, ncols)

# get the size of a single panel in inches
def get_panel_frac(fig):
    # 4. Collect bounding boxes of all "real" subplots
    ax = fig.get_axes()[0]
    bb = ax.get_position()
    xW = bb.x1 - bb.x0
    yH = bb.y1 - bb.y0

    return xW, yH

def check_panel_size(fig, panel_width, panel_height):
    init_width, init_height = fig.get_size_inches()
    xW, yH = get_panel_frac(fig)
    subplot_width = xW * init_width
    subplot_height = yH * init_height
    panel_width_error = abs(panel_width - subplot_width)
    panel_height_error = abs(panel_height - subplot_height)

    if panel_width_error > 0.01 or panel_height_error > 0.01:
        print('Panel width error: ' + str(panel_width_error))
        print('Panel height error: ' + str(panel_height_error))
        return False
    else:
        return True

def round_up_1d(num):
    return math.ceil(num * 10) / 10

def adjust_figure_for_panel_size_auto(fig, panel_width=dPanW, panel_height=dPanH, l=None, r=None, t=None, b=None, ws=None, hs=None):
    """
    Adjust an existing figure so that after tight_layout() the full grid of subplots
    has a total dimension of (ncols * panel_width) x (nrows * panel_height) inches,
    where nrows and ncols are automatically detected.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object (already containing subplots).
    panel_width : float
        Desired width (in inches) of each subplot *panel*.
    panel_height : float
        Desired height (in inches) of each subplot *panel*.
    do_second_tight : bool, optional
        If True, applies tight_layout() again after resizing the figure.
        This can slightly refine the result.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same figure object, but resized so that the total subplot grid
        matches the target dimension (ncols * panel_width x nrows * panel_height).
    """
    # 1. round up panel width and height to 1 decimal place
    panel_width = round_up_1d(panel_width)
    panel_height = round_up_1d(panel_height)


    # 2. Detect the grid shape from the Axes
    nrows, ncols = detect_subplot_grid_shape(fig)
    if nrows == 0 or ncols == 0:
        # Could not detect a valid grid - return unchanged
        return fig

    #desired margins in inches:
    if l is None:
        left = 0.6
    else:
        left = l
    if r is None:
        right = 0.1
    else:
        right = r
    if t is None:
        top = 0.2
    else:
        top = t
    if b is None:
        bottom = 0.3
    else:
        bottom = b
    if ws is None:
        wspace = 0.1
    else:
        wspace = ws

    if hs is None:
        hspace = 0.1
    else:
        hspace = hs

    h_padding = top + bottom + (hspace * (nrows - 1))
    v_padding = left + right + (wspace * (ncols - 1))

    target_fig_size = (ncols * panel_width + v_padding, nrows * panel_height + h_padding)

    #now get the fraction of the figure that will be each padding
    left_frac = left / target_fig_size[0]
    right_frac = right / target_fig_size[0]
    bottom_frac = bottom / target_fig_size[1]
    top_frac = top / target_fig_size[1]
    wspace_frac = wspace/panel_width
    hspace_frac = hspace/panel_height

    #now, set the figure size to the target size
    fig.set_size_inches(target_fig_size)

    #and adjust the margins to the desired values
    fig.subplots_adjust(left=left_frac, right=1-right_frac, top=1-top_frac, bottom=bottom_frac, wspace=wspace_frac, hspace=hspace_frac)

    check = check_panel_size(fig, panel_width, panel_height)
    if check:
        print('Fig size adjusted to: ' + str(fig.get_size_inches()))
        return fig
    else:
        raise ValueError('Panel size not correct after adjustment')


def adjust_figure_for_panel_size_auto_variable_rows(fig, row_heights_inches,
                                                     panel_width=dPanW,
                                                     l=None, r=None, t=None, b=None, hs=None):
    """Like adjust_figure_for_panel_size_auto but for grids with non-uniform row heights.

    row_heights_inches: list of desired heights (inches) for each row, top-to-bottom.
    panel_width: desired width for each column (all columns equal).
    """
    nrows = len(row_heights_inches)
    _, ncols = detect_subplot_grid_shape(fig)
    if ncols == 0:
        return fig

    left   = l  if l  is not None else 0.6
    right  = r  if r  is not None else 0.1
    top    = t  if t  is not None else 0.2
    bottom = b  if b  is not None else 0.3
    hspace = hs if hs is not None else 0.1
    wspace = 0.1

    total_row_h = sum(row_heights_inches)
    h_padding = top + bottom + hspace * (nrows - 1)
    v_padding = left + right + wspace * (ncols - 1)

    fig_w = ncols * panel_width + v_padding
    fig_h = total_row_h + h_padding

    fig.set_size_inches(fig_w, fig_h)

    left_frac   = left   / fig_w
    right_frac  = right  / fig_w
    top_frac    = top    / fig_h
    bottom_frac = bottom / fig_h
    # wspace/hspace as fractions of their respective panel dimension
    avg_row_h = total_row_h / nrows
    wspace_frac = wspace / panel_width
    hspace_frac = hspace / avg_row_h

    fig.subplots_adjust(left=left_frac, right=1 - right_frac,
                        top=1 - top_frac, bottom=bottom_frac,
                        wspace=wspace_frac, hspace=hspace_frac)
    print('Fig size adjusted to: ' + str(fig.get_size_inches()))
    return fig


def adjust_figure_for_panel_size_hetero(fig, panel_width=None, panel_height=None,
                                         l=None, r=None, t=None, b=None, ws=None, hs=None):
    """Adjust figure size for a grid with per-column widths and/or per-row heights.

    panel_width : float or list of float
        Desired width(s) in inches. A scalar applies to all columns; a list
        specifies each column individually (length must equal ncols).
    panel_height : float or list of float
        Desired height(s) in inches. A scalar applies to all rows; a list
        specifies each row individually (length must equal nrows).

    Requires the figure's GridSpec to have matching width_ratios / height_ratios
    (proportional to the values supplied here) so that matplotlib distributes
    space in the right proportions.
    """
    nrows, ncols = detect_subplot_grid_shape(fig)
    if nrows == 0 or ncols == 0:
        return fig

    pw = panel_width  if panel_width  is not None else dPanW
    ph = panel_height if panel_height is not None else dPanH

    col_widths  = [pw] * ncols if not isinstance(pw, (list, tuple)) else list(pw)
    row_heights = [ph] * nrows if not isinstance(ph, (list, tuple)) else list(ph)

    if len(col_widths) != ncols:
        raise ValueError(f'panel_width has {len(col_widths)} entries but figure has {ncols} columns')
    if len(row_heights) != nrows:
        raise ValueError(f'panel_height has {len(row_heights)} entries but figure has {nrows} rows')

    left   = l  if l  is not None else 0.6
    right  = r  if r  is not None else 0.1
    top    = t  if t  is not None else 0.2
    bottom = b  if b  is not None else 0.3
    wspace = ws if ws is not None else 0.1
    hspace = hs if hs is not None else 0.1

    fig_w = sum(col_widths)  + (ncols - 1) * wspace + left + right
    fig_h = sum(row_heights) + (nrows - 1) * hspace + top  + bottom

    fig.set_size_inches(fig_w, fig_h)

    left_frac   = left   / fig_w
    right_frac  = right  / fig_w
    top_frac    = top    / fig_h
    bottom_frac = bottom / fig_h
    # subplots_adjust wspace/hspace are fractions of the mean axis dimension
    mean_col_w  = sum(col_widths)  / ncols
    mean_row_h  = sum(row_heights) / nrows
    wspace_frac = wspace / mean_col_w
    hspace_frac = hspace / mean_row_h

    fig.subplots_adjust(left=left_frac, right=1 - right_frac,
                        top=1 - top_frac, bottom=bottom_frac,
                        wspace=wspace_frac, hspace=hspace_frac)
    print('Fig size adjusted to: ' + str(fig.get_size_inches()))
    return fig