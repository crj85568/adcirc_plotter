import os
import math
import argparse
import cmocean
import geopandas as gpd
import numpy as np
import json
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patheffects
import matplotlib.tri as mtri
from shapely.geometry import box, Point
import cartopy.io.shapereader as shpreader
from typing import Optional, Tuple

# def smart_format(x, _):
#     abs_x = abs(x)
#     if abs_x < 1:
#         return f"{x:,.2f}"
#     elif abs_x < 10:
#         return f"{x:,.1f}"
#     else:
#         return f"{x:,.0f}"
def smart_format(x, _):
    return f"{x:g}"

def generate_balanced_ticks(vmin, vmax, n_ticks=8):
    if vmin >= vmax:
        raise ValueError("vmin must be less than vmax")
    raw_step = (vmax - vmin) / (n_ticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    nice_step = round(raw_step / magnitude) * magnitude
    start = np.ceil(vmin / nice_step) * nice_step
    end = np.floor(vmax / nice_step) * nice_step
    ticks = np.arange(start, end + 0.5 * nice_step, nice_step)
    if ticks[0] > vmin:
        ticks = np.insert(ticks, 0, vmin)
    if ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)
    if len(ticks) > n_ticks:
        idx = np.argsort(np.abs(ticks - 0))[:n_ticks]
        ticks = np.sort(ticks[idx])
    elif len(ticks) < n_ticks:
        while len(ticks) < n_ticks:
            ticks = np.append(ticks, ticks[-1] + nice_step)
    return np.round(ticks, 2)


def set_equal_visual_ticks(cbar, norm, n=8, formatter=None, orientation='vertical'):
    locs = np.linspace(0, 1, n)
    def _inverse(y):
        if hasattr(norm, "inverse"):
            return norm.inverse(y)
        vmin, vcenter, vmax = norm.vmin, norm.vcenter, norm.vmax
        y0 = (vcenter - vmin) / (vmax - vmin)
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        left = y <= y0
        out[left] = vmin + (y[left] / max(y0, np.finfo(float).eps)) * (vcenter - vmin)
        denom = max(1 - y0, np.finfo(float).eps)
        out[~left] = vcenter + ((y[~left] - y0) / denom) * (vmax - vcenter)
        return out
    def _fmt(y, pos):
        val = float(_inverse(np.array([y]))[0])
        return smart_format(val, pos) if formatter is None else formatter(val, pos)
    if orientation == 'vertical':
        cbar.ax.yaxis.set_major_locator(FixedLocator(locs))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    else:
        cbar.ax.xaxis.set_major_locator(FixedLocator(locs))
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    labels = _inverse(locs)
    print(f"Colorbar (equal visual) labels: {labels}")


def meters2deg(distance_meters):
    radius_earth = 6371000 # m
    return (distance_meters * 360.0) / (2 * np.pi * radius_earth)

def deg2meters(distance_deg):
    radius_earth = 6371000 # m
    return (2 * np.pi * radius_earth * distance_deg) / 360.0

def read14(filename="fort.14"):
    print(f"Reading ADCIRC grid from {filename}...")
    nodes = []
    elements = []
    bathymetry = []
    with open(filename, "r") as file:
        header = file.readline().strip()
        counts = file.readline().strip().split()
        n_elements, n_nodes = int(counts[0]), int(counts[1])
        print(f"Found {n_nodes} nodes and {n_elements} elements.")
        for _ in range(n_nodes):
            parts = file.readline().strip().split()
            nodes.append([float(parts[1]), float(parts[2])])
            bathymetry.append(float(parts[3]))
        for _ in range(n_elements):
            parts = file.readline().strip().split()
            elements.append([int(parts[2])-1, int(parts[3])-1, int(parts[4])-1])
    print("Finished reading fort.14 file.")
    return np.array(nodes), np.array(elements), np.array(bathymetry)

def read_adcirc_netcdf(ncfile, variable="zeta_max"):
    print(f"Reading ADCIRC NetCDF file: {ncfile} ...")
    try:
        import netCDF4
    except ImportError:
        raise ImportError("netCDF4 python package is required to read ADCIRC NetCDF output.")
    with netCDF4.Dataset(ncfile, "r") as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        nodes = np.vstack([x, y]).T
        # element is (nfaces, nvertex) 1-based, need 0-based for python
        elements = ds.variables["element"][:].astype(int) - 1
        if elements.shape[1] > 3:
            elements = elements[:, :3]
        # remove elements with invalid node indices
        elements = elements[(elements >= 0).all(axis=1)]
        # get the variable to plot
        if variable in ds.variables:
            values = ds.variables[variable][:]
            # If there's a fill value, mask it
            if hasattr(ds.variables[variable], "_FillValue"):
                fv = ds.variables[variable]._FillValue
                values = np.where(values == fv, np.nan, values)
        else:
            raise ValueError(f"Variable '{variable}' not found in NetCDF file.")
    print(f"Read {nodes.shape[0]} nodes, {elements.shape[0]} elements, variable '{variable}'")
    return nodes, elements, values


def plot_adcirc_mesh(
    nodes,
    elements,
    values=None,
    title: str = "ADCIRC Grid Plot",
    add_basemap: bool = False,
    show_state_labels: bool = False,
    show_state_boundaries: bool = False,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_file: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    shapefile_overlay: Optional[str] = None,
    shapefile_overlay_label: Optional[str] = None,
    clabel: Optional[str] = None,
    multiplier: float = 1.0,
    draw_mesh: bool = False,
    tick_interval: Optional[float] = None,
    transparency: float = 1.0,
    no_colorbar=False,
    linewidth: float=0.12,
    levels: Optional[list] = None,
    binned_colorbar: bool=False
):
    fig, ax = init_figure()
    vals = preprocess_values(values, multiplier)
    bnds = validate_bounds(bounds)
    triang = setup_triangulation(nodes, elements)
    vmin, vmax, cmap = prepare_color(vals, vmin, vmax, cmap)
    # UPDATE THIS CALL:
    tpc, norm = draw_tripcolor(ax, triang, vals, vmin, vmax, cmap, transparency, levels=levels)
    
    if draw_mesh:
        add_mesh_lines(ax, triang, linewidth)
        
    if tpc and not no_colorbar:
        # UPDATE THIS CALL:
        set_colorbar(fig, ax, tpc, cmap, norm, vmin, vmax, clabel, transparency, 
                     tick_interval, levels=levels, binned=binned_colorbar)
    render_overlays(ax, nodes, bnds, add_basemap, show_state_labels,
                    show_state_boundaries, shapefile_overlay, shapefile_overlay_label)
    finalize(fig, ax, title, output_file)


def preprocess_values(values, multiplier: float):
    if values is None:
        return None
    clean = np.where(values == -99999, np.nan, values)
    return clean * multiplier


def validate_bounds(bounds):
    bnds, changed, swapped = normalize_bounds(bounds)
    if changed:
        print(f"Corrected bounds -> {bnds} (swapped lat/lon: {swapped})")
    return bnds


def setup_triangulation(nodes, elements):
    print("Preparing triangulation...")
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)


def init_figure():
    print("Generating plot...")
    return plt.subplots(figsize=(10, 8))


def prepare_color(values, vmin, vmax, cmap):
    if isinstance(vmin, (int, float)):
        vmin = float(vmin)
    if isinstance(vmax, (int, float)):
        vmax = float(vmax)
    if values is not None:
        dmin, dmax = np.nanmin(values), np.nanmax(values)
        vmin = dmin if vmin is None else vmin
        vmax = dmax if vmax is None else vmax
        print(f"Color scale bounds: vmin={vmin}, vmax={vmax}")
    return vmin, vmax, resolve_cmap(cmap)


def resolve_cmap(cmap):
    if not (isinstance(cmap, str) and cmap.startswith("cmocean.")):
        return cmap
    name = cmap.split(".", 1)[1]
    flip = name.endswith("_r")
    name = name[:-2] if flip else name
    try:
        import cmocean  # lazy import
        cm = getattr(cmocean.cm, name, None)
    except Exception as exc:
        raise ValueError("cmocean requested but not available") from exc
    if cm is None:
        raise ValueError(f"Unknown cmocean colormap: {name}")
    return cm.reversed() if flip else cm


def draw_tripcolor(ax, triang, values, vmin, vmax, cmap, alpha, levels=None):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if values is None:
        return None, None
    # 1. PIECEWISE LINEAR MAPPING (The "Smooth Binned" Look)
    if levels is not None:
        # Sort levels just in case
        levels = sorted(levels)
        n_levels = len(levels)
        
        # Create a copy of values to transform
        # We map the real values to their "Index" in the levels list.
        # e.g., if levels are [-60, -30, ...], a value of -45 becomes 0.5 (halfway through first interval)
        values_transformed = np.zeros_like(values)
        
        # Vectorized interpolation to 'Level Space'
        # This gives equal visual weight to every interval regardless of numeric size
        values_transformed = np.interp(values, levels, np.arange(n_levels))
        
        # Plot the TRANSFORMED data (0 to n_levels-1)
        # We use standard Gouraud shading on this transformed data
        tpc = ax.tripcolor(triang, values_transformed, cmap=cmap, 
                           vmin=0, vmax=n_levels-1,
                           shading="gouraud", alpha=alpha, zorder=10)
        
        # Return the levels as 'norm' so the colorbar knows what to do
        return tpc, levels
    # if levels is not None:
    #     import matplotlib.colors as mcolors
    #     norm = mcolors.BoundaryNorm(levels, cmap.N, extend='both')
    #     tpc = ax.tripcolor(triang, values, cmap=cmap, norm=norm, shading='gouraud',
    #                        alpha=alpha, zorder=10)
    #     return tpc, norm
    values = np.clip(values, vmin, vmax)
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        tpc = ax.tripcolor(triang, values, cmap=cmap, norm=norm, shading="gouraud",
                           alpha=alpha, zorder=10)
    else:
        norm = None
        tpc = ax.tripcolor(triang, values, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="gouraud", alpha=alpha, edgecolors="none", zorder=10)
    return tpc, norm


def add_mesh_lines(ax, triang):
    print("Drawing mesh triangles...")
    ax.triplot(triang, color="white", linewidth=0.2, alpha=1, zorder=11)

def set_colorbar(fig, ax, tpc, cmap, norm, vmin, vmax, clabel, alpha, tick_interval, levels=None, binned=False):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    # MODE A: Binned (Discrete Blocks)
    # We create a fake "mappable" just for the legend so it looks like blocks
    if levels is not None and binned:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Create the discrete norm
        cbar_norm = mcolors.BoundaryNorm(levels, cmap.N, extend='both')
        
        # Create a dummy object (not plotted)
        mappable = cm.ScalarMappable(norm=cbar_norm, cmap=cmap)
        mappable.set_array([]) 
        
        cbar = fig.colorbar(mappable, cax=cax, orientation='vertical', alpha=alpha, extend='both')
        
        # Force ticks to match the levels
        cbar.set_ticks(levels)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))

    # MODE B: Smooth (Gradient) - Default
    # We use the actual plot object ('tpc') so the bar matches the figure exactly
    else:
        # Note: If using the "Piecewise Linear" draw_tripcolor, 'norm' here is actually the levels list
        # We pass it through so configure_cbar_ticks can label it correctly
        cbar = make_colorbar(fig, cax, tpc, cmap, norm, vmin, vmax, clabel, alpha)
        configure_cbar_ticks(cbar, norm, vmin, vmax, tick_interval, levels)

    if clabel:
        cbar.set_label(clabel)
# def set_colorbar(fig, ax, tpc, cmap, norm, vmin, vmax, clabel, alpha,
#                  tick_interval, levels=None):
#     cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.05)
#     cbar = make_colorbar(fig, cax, tpc, cmap, norm, vmin, vmax, clabel, alpha)
#     configure_cbar_ticks(cbar, norm, vmin, vmax, tick_interval, levels)
# def set_colorbar(fig, ax, tpc, cmap, norm, vmin, vmax, clabel, alpha, tick_interval, levels=None):
#     if isinstance(cmap, str):
#         cmap = plt.get_cmap(cmap)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2%", pad=0.05)

#     # --- THE TRICK: Decouple Plot from Colorbar ---
#     if levels is not None:
#         import matplotlib.cm as cm
#         import matplotlib.colors as mcolors
        
#         # 1. Create a Discrete Norm just for the Colorbar
#         #    (This forces the bar to look like solid blocks)
#         cbar_norm = mcolors.BoundaryNorm(levels, cmap.N, extend='both')
        
#         # 2. Create a "Dummy" Mappable
#         #    (We don't plot this, we just use it to generate the bar)
#         mappable = cm.ScalarMappable(norm=cbar_norm, cmap=cmap)
#         mappable.set_array([]) # Fake data
        
#         # 3. Build the Colorbar using the Dummy
#         cbar = fig.colorbar(mappable, cax=cax, orientation='vertical', alpha=alpha, extend='both')
        
#         # 4. Set ticks exactly at the level boundaries
#         cbar.set_ticks(levels)
#         cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
        
#     else:
#         # Standard behavior (Linked 1-to-1)
#         cbar = make_colorbar(fig, cax, tpc, cmap, norm, vmin, vmax, clabel, alpha)
#         configure_cbar_ticks(cbar, norm, vmin, vmax, tick_interval, levels)

#     # Label the bar
#     if clabel:
#         cbar.set_label(clabel)

def make_colorbar(fig, cax, tpc, cmap, norm, vmin, vmax, clabel, alpha):
    if alpha is not None and alpha < 1:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        sm = cm.ScalarMappable(cmap=cmap,
                               norm=norm or mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        return fig.colorbar(sm, cax=cax, label=clabel, extend="both", aspect=40)
    return fig.colorbar(tpc, cax=cax, label=clabel, extend="both", aspect=40)


def configure_cbar_ticks(cbar, norm, vmin, vmax, tick_interval, levels=None):
    if levels is not None and isinstance(levels, list):
        # The data is plotted as 0, 1, 2, 3...
        # So we put ticks exactly at those integers
        n_levels = len(levels)
        ticks_indices = np.arange(n_levels)
        cbar.set_ticks(ticks_indices)
        
        # And we label them with the ORIGINAL level values
        cbar.ax.set_yticklabels([smart_format(x, None) for x in levels])
        return
    if norm is not None:
        n_total = 10
        n_side = (n_total - 1) // 2
        ticks = np.r_[np.linspace(vmin, 0, n_side + 1)[:-1], 0,
                      np.linspace(0, vmax, n_side + 1)[1:]]
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
        print("Colorbar ticks (equal numeric spacing per side):", ticks)
        return
    if tick_interval and vmin is not None and vmax is not None:
        nticks = int((vmax - vmin) / tick_interval) + 1
        ticks = np.linspace(vmin, vmax, nticks)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
        print(f"Colorbar ticks (custom interval {tick_interval}):", ticks)
    else:
        cbar.locator = mticker.MaxNLocator(nbins=8)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
        print(f"Colorbar: vmin={vmin}, vmax={vmax}, norm=None")


def render_overlays(ax, nodes, bounds, add_basemap, show_labels, show_edges,
                    shp_path, shp_label):
    set_bounds(ax, nodes, bounds)
    if shp_path:
        plot_shapefile_overlay(ax, shp_path, shp_label)
    if add_basemap:
        add_basemap_layer(ax, bounds)
    if add_basemap and (show_labels or show_edges):
        add_state_overlays(ax, bounds, show_labels, show_edges)


def set_bounds(ax, nodes, bounds):
    if bounds:
        print(f"Setting bounding box to: {bounds}")
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    else:
        ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
        ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())


def add_basemap_layer(ax, bounds):
    print("Adding basemap...")
    ctx.add_basemap(ax,
                    crs="EPSG:4326",
                    source=ctx.providers.Esri.WorldImagery,
                    reset_extent=False,
                    attribution=False,
                    zorder=0)
    if bounds:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])


def plot_shapefile_overlay(ax, path, label):
    print(f"Plotting shapefile overlay from {path}...")
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            print("Warning: shapefile has no features.")
            return
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
        if gdf.crs is None:
            print("Warning: shapefile CRS not defined. Assuming EPSG:4326.")
        gdf.plot(ax=ax,
                 edgecolor="black",
                 facecolor="none",
                 linewidth=2,
                 zorder=20,
                 label=label or None)
    except Exception as exc:
        print(f"Warning: Failed to load shapefile overlay: {exc}")


def add_state_overlays(ax, bounds, show_labels, show_edges):
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box
        from matplotlib import patheffects

        print("Adding state overlay...")
        rdr = shpreader.Reader(
            shpreader.natural_earth("10m", "cultural", "admin_1_states_provinces")
        )
        label_box = box(*bounds) if bounds else None
        for rec in rdr.records():
            plot_state(ax, rec, show_edges, show_labels, label_box, patheffects)
    except Exception as exc:
        print(f"Warning: could not render state overlay: {exc}")


def plot_state(ax, rec, show_edges, show_labels, label_box, patheffects):
    geom, name = rec.geometry, rec.attributes.get("name")
    if geom.is_empty:
        return
    if show_edges:
        plot_geom_edges(ax, geom)
    if show_labels and not geom.centroid.is_empty:
        if label_box and not label_box.contains(geom.centroid):
            return
        ax.text(geom.centroid.x,
                geom.centroid.y,
                name,
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                path_effects=[patheffects.withStroke(linewidth=2,
                                                     foreground="black")],
                zorder=10)


def plot_geom_edges(ax, geom):
    try:
        ax.plot(*geom.exterior.xy, color="white", linewidth=0.5, zorder=5)
    except Exception:
        for part in getattr(geom, "geoms", []):
            ax.plot(*part.exterior.xy, color="white", linewidth=0.5, zorder=5)


def finalize(fig, ax, title: Optional[str], output_file: Optional[str]):
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    dedupe_legend(ax)
    fig.tight_layout()
    if output_file:
        print(f"Saving plot to {output_file}...")
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)
        print("Plot saved.")
    else:
        print("Displaying plot interactively...")
        plt.show()


def dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen, kept_h, kept_l = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            kept_h.append(h)
            kept_l.append(l)
            seen.add(l)
    if kept_l:
        leg = ax.legend(kept_h, kept_l, loc="upper left",
                        framealpha=0.9, fancybox=True)
        leg.set_zorder(1000)

def normalize_bounds(bounds):
    if not bounds or len(bounds) != 4:
        return None, False, False
    x0, y0, x1, y1 = map(float, bounds)
    swapped = (abs(x0) <= 90 and abs(y0) > 90) or (abs(x1) <= 90 and abs(y1) > 90)
    if swapped:
        x0, y0, x1, y1 = y0, x0, y1, x1
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    changed = swapped or (xmin != x0) or (ymin != y0)
    return (xmin, ymin, xmax, ymax), changed, swapped




def animate_adcirc_zeta(ncfile, variable='zeta',
    output_file="animation.mp4",
    fps=10,
    start_index=0,
    end_index=None,
    step=1,
    bounds=None,
    vmin=None,
    vmax=None,
    cmap="RdYlBu_r",
    add_basemap=False,
    shapefile_overlay=None,
    shapefile_overlay_label=None,
    clabel=None,
    show_state_labels=False,
    show_state_boundaries=False,
    multiplier=1.0,
    draw_mesh=False,
    transparency=1.0
):
    """
    Create an animation from a time-varying ADCIRC NetCDF (e.g., fort.63.nc) with `<variable>(time,node)`, default variable='zeta'.
    Saves MP4 (via ffmpeg) or GIF (via pillow) depending on output_file extension.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError("netCDF4 python package is required.")

    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    import numpy as np
    import datetime as _dt

    print(f"Opening {ncfile} ...")
    ds = netCDF4.Dataset(ncfile, "r")

    x = ds.variables["x"][:]
    y = ds.variables["y"][:]
    nodes = np.vstack([x, y]).T
    elements = ds.variables["element"][:].astype(int) - 1
    if elements.shape[1] > 3:
        elements = elements[:, :3]

    time_var = ds.variables.get("time")
    if time_var is None:
        ds.close()
        raise ValueError("NetCDF has no 'time' variable.")
    times_num = time_var[:]
    try:
        from netCDF4 import num2date
        times_dt = num2date(times_num, time_var.units)
    except Exception:
        base = getattr(time_var, "base_date", None) or getattr(time_var, "units", "seconds since 2000-01-01 00:00:00")
        from netCDF4 import num2date
        times_dt = num2date(times_num, base)

    var = ds.variables.get(variable)
    if var is None or var.ndim != 2:
        ds.close()
        raise ValueError(f"Expected '{variable}(time,node)' in the file.")
    zeta = var
    ntime = zeta.shape[0]
    if end_index is None or end_index >= ntime:
        end_index = ntime - 1
    idxs = np.arange(start_index, end_index + 1, step, dtype=int)
    print(f"Animating {len(idxs)} frames from indices [{idxs[0]}..{idxs[-1]}] (step={step}).")

    # Precompute triangulation once
    triang = tri.Triangulation(nodes[:,0], nodes[:,1], elements)

    # Determine vmin/vmax if not provided
    if vmin is None or vmax is None:
        # Sample a few frames for robust limits
        sample_idxs = np.unique(np.clip(np.linspace(0, ntime-1, 8, dtype=int), 0, ntime-1))
        sample = np.hstack([var[i,:] for i in sample_idxs])
        sample = np.where(sample == -99999, np.nan, sample) * multiplier
        if vmin is None: vmin = np.nanpercentile(sample, 2)
        if vmax is None: vmax = np.nanpercentile(sample, 98)
        if vmin == vmax:
            vmin, vmax = vmin - 0.1, vmax + 0.1
        print(f"Auto color limits (based on sampled frames): vmin={vmin:.3f}, vmax={vmax:.3f}")

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Apply bounds normalization/validation (re-using helper if present)
    try:
        bnds, changed, swapped = normalize_bounds(bounds)
        bounds = bnds
        if changed:
            print(f"Corrected bounds -> {bounds} (swapped lat/lon: {swapped})")
    except Exception:
        pass

    # Initial frame
    vals0 = var[idxs[0], :]
    vals0 = np.where(vals0 == -99999, np.nan, vals0) * multiplier

    tpc = ax.tripcolor(triang, vals0, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    if bounds:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    # Optional mesh overlay
    if draw_mesh:
        ax.triplot(triang, linewidth=0.1, color="k", alpha=0.2)

    # Basemap / overlays using existing helpers from plot_adcirc_mesh (if callable)
    try:
        if add_basemap:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Stamen.TerrainBackground, attribution="")
        if shapefile_overlay:
            gdf = gpd.read_file(shapefile_overlay)
            gdf.boundary.plot(ax=ax, linewidth=0.5, color='k')
            if shapefile_overlay_label:
                add_label_column(gdf, label_field=shapefile_overlay_label, ax=ax)
        if show_state_boundaries or show_state_labels:
            add_state_boundaries(ax, show_labels=show_state_labels)
    except Exception as e:
        print(f"Overlay warning: {e}")

    # Colorbar
    from matplotlib import cm as _cm, colors as _colors
    norm_cb = _colors.Normalize(vmin=vmin, vmax=vmax)
    sm = _cm.ScalarMappable(cmap=cmap, norm=norm_cb)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=clabel if clabel else None, extend='both', aspect=40)
    if bounds:
        try:
            set_equal_visual_ticks(cbar, norm_cb, n=8, orientation='vertical')
        except Exception:
            pass

    title_txt = ax.set_title(f"{getattr(ds, 'title', 'ADCIRC')} — {str(times_dt[idxs[0]])}")

    def update(i):
        vals = var[i, :]
        vals = np.where(vals == -99999, np.nan, vals) * multiplier
        # Update the face colors in the QuadMesh generated by tripcolor (for 'flat' shading it's stored in ._A)
        tpc.set_array(vals[elements].mean(axis=1)) if False else tpc.set_array(vals)  # fallback direct set
        # Easiest reliable method: re-draw tripcolor by updating the array data
        tpc.set_clim(vmin, vmax)
        title_txt.set_text(f"{getattr(ds, 'title', 'ADCIRC')} — {str(times_dt[i])}")
        return (tpc, title_txt)

    anim = FuncAnimation(fig, update, frames=idxs, blit=False, interval=1000//max(fps,1))

    # Pick writer based on extension
    ext = os.path.splitext(output_file)[1].lower()
    if ext in [".mp4", ".m4v", ".mov"]:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800, metadata=dict(artist="adcirc_plotter"))
        except Exception as e:
            print(f"FFmpeg not available ({e}); falling back to GIF.")
            ext = ".gif"
    if ext == ".gif":
        writer = PillowWriter(fps=fps)
    if ext not in [".gif", ".mp4", ".m4v", ".mov"]:
        # default to mp4
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800, metadata=dict(artist="adcirc_plotter"))
            ext = ".mp4"
        except Exception:
            writer = PillowWriter(fps=fps)
            ext = ".gif"
            output_file = os.path.splitext(output_file)[0] + ext

    print(f"Saving animation to {output_file} ...")
    anim.save(output_file, writer=writer, dpi=150)
    plt.close(fig)
    ds.close()
    print("Animation saved.")


def is_netcdf_file(filename):
    return filename.lower().endswith(".nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ADCIRC fort.14 or NetCDF output using config file or CLI arguments."
    )

    parser.add_argument("--animate", action="store_true", help="Create an animation from time-varying 'zeta(time,node)' (fort.63.nc)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the animation")
    parser.add_argument("--start-index", type=int, default=0, help="Start time index for animation")
    parser.add_argument("--end-index", type=int, default=None, help="End time index for animation (inclusive)")
    parser.add_argument("--step", type=int, default=1, help="Step between time indices")
    parser.add_argument("--format", type=str, default=None, help="Force output format: mp4 or gif (by default uses output file extension)")

    parser.add_argument("--config", type=str, help="Path to JSON configuration file (overrides other CLI options)")
    parser.add_argument("--input", type=str, help="Path to fort.14 or NetCDF file")
    parser.add_argument("--output", type=str, help="Output PNG file path")
    parser.add_argument("--vmin", type=float, help="Minimum value for color scale")
    parser.add_argument("--vmax", type=float, help="Maximum value for color scale")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=('xmin', 'ymin', 'xmax', 'ymax'), help="Bounding box (xmin ymin xmax ymax)")
    parser.add_argument("--no-basemap", action="store_true", help="Disable basemap overlay")
    parser.add_argument("--cmap", type=str, help="Colormap name or 'cmocean.deep'")
    parser.add_argument("--shapefile-overlay", type=str, help="Path to shapefile overlay")
    parser.add_argument("--clabel", type=str, help="Colorbar label")
    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Value multiplier (e.g., 3.28084 for meters to feet)")
    parser.add_argument("--variable", type=str, default="zeta_max", help="NetCDF variable to plot (e.g., zeta_max, depth)")
    parser.add_argument("--draw-mesh", action="store_true", help="Draw mesh triangles on the plot")
    parser.add_argument("--show-state-labels", action="store_true", help="Show state labels on the plot")
    parser.add_argument("--show-state-boundaries", action="store_true", help="Show state boundaries on the plot")
    parser.add_argument("--tick-interval", type=float, help="Custom tick interval for colorbar (if not using vmin/vmax)")
    parser.add_argument("--transparency", type=float, default=1.0, help="Transparency level for the colormap (0.0 to 1.0)")
    parser.add_argument("--no_colorbar", action="store_true", help="Disable colorbar")
    parser.add_argument("--fontsize", type=int, default=14, help="Base font size for the plot")
    parser.add_argument("--levels", nargs="+", type=float, help="List of specific color labels")
    parser.add_argument('--binned-colorbar', action='store_true', help='Render the colorbar with normalized bins')
    
    args = parser.parse_args()
    plt.rcParams.update({'font.size': args.fontsize})
    
    # Load config if provided, then let CLI override
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    
    # Helper to pick output format
    if args.format:
        if args.output:
            root, _ext = os.path.splitext(args.output)
            args.output = root + (".mp4" if args.format.lower() == "mp4" else ".gif")
    
    if args.animate:
        if not args.input:
            raise SystemExit("Please provide --input pointing to a time-varying ADCIRC NetCDF (e.g., fort.63.nc).")
        out = args.output or "animation.mp4"
        animate_adcirc_zeta(
            ncfile=args.input,
            output_file=out,
            fps=args.fps,
            start_index=args.start_index,
            end_index=args.end_index,
            step=args.step,
            bounds=config.get("bbox"),
            vmin=args.vmin if args.vmin is not None else config.get("vmin"),
            vmax=args.vmax if args.vmax is not None else config.get("vmax"),
            cmap=args.cmap if hasattr(args, "cmap") and args.cmap else config.get("cmap", "RdYlBu_r"),
            add_basemap=not args.no_basemap if hasattr(args, "no_basemap") else config.get("basemap", False),
            shapefile_overlay=config.get("shapefile_overlay"),
            shapefile_overlay_label=config.get("shapefile_overlay_label"),
            clabel=(config.get("clabel") if config.get("clabel") is not None else f"{(args.variable or config.get('variable','zeta'))} (units)") ,
            show_state_labels=config.get("show_state_labels", False),
            show_state_boundaries=config.get("show_state_boundaries", False),
            multiplier=config.get("multiplier", 1.0),
            draw_mesh=config.get("draw_mesh", False),
            transparency=config.get("transparency", 1.0),
            levels=config.get('levels', None)
        )
        raise SystemExit(0)
    
    
    if args.config:
        print(f"Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = vars(args)

    print("Starting ADCIRC plotter...")
    input_file = args.input or config.get("input", "fort.14")
    variable = config.get("variable", "zeta_max")
    if is_netcdf_file(input_file):
        nodes, elements, values = read_adcirc_netcdf(input_file, variable)
    else:
        nodes, elements, bathy = read14(input_file)
        # Only fort.14: always bathymetry
        values = -1*bathy

    plot_adcirc_mesh(
        nodes, elements, values,
        title=config.get("title"),
        add_basemap=not config.get("no_basemap", False),
        bounds=config.get("bbox"),
        vmin=config.get("vmin"),
        vmax=config.get("vmax"),
        output_file=(args.output if hasattr(args, 'output') and args.output else config.get("output")),
        cmap=config.get("cmap", "RdYlBu"),
        shapefile_overlay=config.get("shapefile_overlay"),
        shapefile_overlay_label=config.get("shapefile_overlay_label"),
        clabel=config.get("clabel"),
        show_state_labels=config.get("show_state_labels", False),
        show_state_boundaries=config.get("show_state_boundaries", False),
        multiplier=config.get("multiplier", 1.0),
        draw_mesh=config.get("draw_mesh", True),
        tick_interval=config.get("tick_interval", None),  # New parameter for custom tick intervals,
        transparency=config.get("transparency", 1.0),  # New parameter for transparency
        no_colorbar=config.get('no_colorbar', False),
        levels=config.get('levels', None),
        binned_colorbar=args.binned_colorbar
    )
    print("Done.")