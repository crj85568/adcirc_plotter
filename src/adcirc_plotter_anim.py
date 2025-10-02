from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import argparse
import os
import json
import cmocean
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from matplotlib import patheffects
import math
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

def smart_format(x, _):
    abs_x = abs(x)
    if abs_x < 1:
        return f"{x:,.2f}"
    elif abs_x < 10:
        return f"{x:,.1f}"
    else:
        return f"{x:,.0f}"

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

def normalize_bounds(bounds):
    """
    Ensure bounds are [xmin, ymin, xmax, ymax] where x=longitude (-180..180), y=latitude (-90..90).
    Fix reversed min/max; swap if given as (lat, lon, lat, lon); clamp to valid ranges.
    Returns (fixed_bounds, changed: bool, swapped: bool).
    """
    if bounds is None:
        return bounds, False, False
    if len(bounds) != 4:
        print("Warning: bounds must have 4 elements; ignoring provided bounds.")
        return None, False, False
    try:
        x1, y1, x2, y2 = map(float, bounds)
    except Exception:
        print("Warning: bounds not numeric; ignoring provided bounds.")
        return None, False, False

    # Detect (lat,lon,lat,lon): x look like lat; y look like lon (at least one |lon| > 90)
    x_like_lat = (abs(x1) <= 90) and (abs(x2) <= 90)
    y_like_lon = (abs(y1) <= 180) and (abs(y2) <= 180) and ((abs(y1) > 90) or (abs(y2) > 90))
    swapped = False
    if x_like_lat and y_like_lon:
        x1, y1, x2, y2 = y1, x1, y2, x2
        swapped = True
        print("Notice: bounds appeared as (lat, lon, lat, lon); corrected to (lon, lat, lon, lat).")

    changed = swapped

    # Fix reversed min/max
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    if (xmin != x1) or (xmax != x2) or (ymin != y1) or (ymax != y2):
        changed = True
        print("Notice: bounds min/max were reversed; corrected to [xmin, ymin, xmax, ymax].")

    # Clamp to valid geographic ranges
    xmin = max(-180.0, min(180.0, xmin))
    xmax = max(-180.0, min(180.0, xmax))
    ymin = max(-90.0,  min(90.0,  ymin))
    ymax = max(-90.0,  min(90.0,  ymax))

    if xmin == xmax or ymin == ymax:
        print("Warning: bounds collapsed to zero area; ignoring provided bounds.")
        return None, changed, swapped

    return [xmin, ymin, xmax, ymax], changed, swapped


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

def plot_adcirc_mesh(nodes, elements, values=None, title="ADCIRC Grid Plot", add_basemap=False,
                     show_state_labels=False, show_state_boundaries=False,
                     bounds=None, vmin=None, vmax=None, output_file=None,
                     cmap="RdYlBu_r", shapefile_overlay=None, shapefile_overlay_label=None,
                     clabel=None, multiplier=1.0, draw_mesh=False, tick_interval=None, transparency=1.0):
    if values is not None:
        values = np.where(values == -99999, np.nan, values)
    
        # Validate and correct bounds if provided
    bounds, _bounds_changed, _bounds_swapped = normalize_bounds(bounds)
    if _bounds_changed:
        print(f"Corrected bounds -> {bounds} (swapped lat/lon: {_bounds_swapped})")
    
    print("Preparing triangulation...")
    triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin = float(vmin) if vmin is not None else None
    vmax = float(vmax) if vmax is not None else None
    
    if values is not None:
        values = values * multiplier
        data_min = np.nanmin(values)
        data_max = np.nanmax(values)
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
        print(f"Color scale bounds: vmin={vmin}, vmax={vmax}")
        if cmap.startswith("cmocean."):
            parts = cmap.split(".")
            cmap_name = parts[1]
            flip = cmap_name.endswith("_r")
            if flip:
                cmap_name = cmap_name[:-2]
            cmap = getattr(cmocean.cm, cmap_name, None)
            if cmap is None:
                raise ValueError(f"Unknown cmocean colormap: {cmap_name}")
            if flip:
                cmap = cmap.reversed()
        # Clip values to within [vmin, vmax]
        values = np.clip(values, vmin, vmax)
        # ---- create the mappable ----
        if vmin < 0 < vmax:
            levels = np.linspace(vmin, vmax, 101)
            if levels[-1] < vmax:
                levels = np.append(levels, vmax)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            tpc = ax.tripcolor(
                triang,
                values,
                cmap=cmap,
                norm=norm,
                shading='gouraud',
                alpha=transparency,
                zorder=10
            )
        else:
            norm = None
            levels = np.linspace(vmin, vmax, 101, endpoint=True)
            tpc = ax.tripcolor(
                triang,
                values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading='gouraud',
                alpha=transparency,
                edgecolors='none',
                zorder=10
            )
        if draw_mesh:
            print("Drawing mesh triangles...")
            ax.triplot(
                triang,
                color='black',
                linewidth=0.12,
                alpha=0.75,
                zorder=11
            )
        # ---- colorbar & ticks ----
       
        # Create a colorbar axis that matches the main axis height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)  # 5% of axis width, small gap

        if transparency is not None and transparency < 1:
            # Use a ScalarMappable (opaque) for the colorbar
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            norm_cb = norm if norm is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm_cb)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, label=clabel if clabel else None, extend='both', aspect=40)
        else:
            # Use the plotted object itself for the colorbar
            cbar = fig.colorbar(tpc, cax=cax, label=clabel if clabel else None, extend='both', aspect=40)
        

        if norm is not None:
            n_total = 10
            n_side = (n_total - 1) // 2
            pos_ticks = np.linspace(0, vmax, n_side + 1)[1:]
            neg_ticks = np.linspace(vmin, 0, n_side + 1)[:-1]
            ticks = np.concatenate([neg_ticks, [0], pos_ticks])
            cbar.set_ticks(ticks)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
            print(f"Colorbar ticks (equal numeric spacing per side): {ticks}")
        else:
            if tick_interval is not None and vmin is not None and vmax is not None:
                # Use user-specified tick interval
                nticks = int((vmax - vmin) / tick_interval) + 1
                ticks = np.linspace(vmin, vmax, nticks)
                cbar.set_ticks(ticks)
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
                print(f"Colorbar ticks (custom interval {tick_interval}): {ticks}")
            else:
                cbar.locator = ticker.MaxNLocator(nbins=8)
                cbar.update_ticks()
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
                print(f"Colorbar: vmin={vmin}, vmax={vmax}, norm=None")
            
        if shapefile_overlay:
            print(f"Plotting shapefile overlay from {shapefile_overlay}...")
            try:
                gdf = gpd.read_file(shapefile_overlay)
                if gdf.empty:
                    print("Warning: shapefile has no features.")
                else:
                    if gdf.crs is not None:
                        gdf = gdf.to_crs("EPSG:4326")
                    else:
                        print("Warning: shapefile CRS not defined. Assuming EPSG:4326.")
                    gdf.plot(
                        ax=ax,
                        edgecolor='black',
                        facecolor='none',
                        linewidth=2,
                        zorder=20,
                        label=shapefile_overlay_label if shapefile_overlay_label else None
                    )
            except Exception as e:
                print(f"Warning: Failed to load shapefile overlay: {e}")
    if bounds:
        print(f"Setting bounding box to: {bounds}")
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    else:
        ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
        ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    if add_basemap:
        print("Adding basemap...")
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery,
                        reset_extent=False, attribution=False, zorder = 0)
        if bounds:
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
    if add_basemap and (show_state_labels or show_state_boundaries):
        try:
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import box
            from matplotlib import patheffects
            print("Adding state overlay...")
            states_shp = shpreader.natural_earth(resolution='10m', category='cultural',
                                                name='admin_1_states_provinces')
            reader = shpreader.Reader(states_shp)
            states = list(reader.records())
            label_box = box(*bounds) if bounds else None
            for state in states:
                geom = state.geometry
                name = state.attributes['name']
                if geom.is_empty:
                    continue
                if show_state_boundaries:
                    try:
                        ax.plot(*geom.exterior.xy, color='white', linewidth=0.5, zorder=5)
                    except:
                        for part in geom.geoms:
                            ax.plot(*part.exterior.xy, color='white', linewidth=0.5, zorder=5)
                if show_state_labels and not geom.centroid.is_empty:
                    if label_box and not label_box.contains(geom.centroid):
                        continue
                    x, y = geom.centroid.x, geom.centroid.y
                    ax.text(
                        x, y, name,
                        fontsize=8,
                        ha='center',
                        va='center',
                        color='white',
                        path_effects=[patheffects.withStroke(linewidth=2, foreground='black')],
                        zorder=10
                    )
        except Exception as e:
            print(f"Warning: could not render state overlay: {e}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc='upper right')
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    H, L = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            H.append(h); L.append(l); seen.add(l)
    if L:
        leg = ax.legend(H, L,
                        loc='upper left',
                        framealpha=0.9, fancybox=True)
        leg.set_zorder(1000)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    if output_file:
        print(f"Saving plot to {output_file}...")
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)
        print("Plot saved.")
    else:
        print("Displaying plot interactively...")
        plt.show()
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

    tpc = ax.tripcolor(triang, vals0, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)
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

    
    args = parser.parse_args()
    
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
            transparency=config.get("transparency", 1.0)
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
        values = bathy

    plot_adcirc_mesh(
        nodes, elements, values,
        title=config.get("title"),
        add_basemap=not config.get("no_basemap", False),
        bounds=config.get("bbox"),
        vmin=config.get("vmin"),
        vmax=config.get("vmax"),
        output_file=(args.output if hasattr(args, 'output') and args.output else config.get("output")),
        cmap=config.get("cmap", "RdYlBu_r"),
        shapefile_overlay=config.get("shapefile_overlay"),
        shapefile_overlay_label=config.get("shapefile_overlay_label"),
        clabel=config.get("clabel"),
        show_state_labels=config.get("show_state_labels", False),
        show_state_boundaries=config.get("show_state_boundaries", False),
        multiplier=config.get("multiplier", 1.0),
        draw_mesh=config.get("draw_mesh", False),
        tick_interval=config.get("tick_interval", None),  # New parameter for custom tick intervals,
        transparency=config.get("transparency", 1.0)  # New parameter for transparency
    )
    print("Done.")