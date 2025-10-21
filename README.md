# adcplot

High-quality maps and animations for **ADCIRC** (and PTM, in progress) that work from the **command line** and **Jupyter notebooks**. Supports scalar fields (`zeta`, `zeta_max`, `vel_max`, `u`, `v`) and vectors, with optional **Cartopy coastlines** or **ESRI web tiles** as basemaps. Designed for portability (like figuregen), testability, and performance (parallel frame rendering + static colorbars).

---

## Features

- **Inputs**
  - ADCIRC NetCDF: `fort.63.nc`, `fort.64.nc`, `fort.74.nc`
  - Mesh: `fort.14` (optional, for topology)
  - PTM HDF5/XMDF (adapter scaffolding; parity WIP)

- **Outputs**
  - Static maps (PNG)
  - Animations (MP4 / GIF) with fixed-size frames (no “breathing”)
  - Scalar and vector plots (quiver; streamline on plain Matplotlib)

- **Rendering**
  - Backend: Matplotlib (default). PyGMT skeleton optional.
  - Basemap: `none`, `cartopy` (NE/GSHHS), or `tiles` (ESRI imagery/topo/streets)
  - Smooth (`tricontourf`) on projected axes; **true flat** per-triangle shading with `--shading flat`
  - **Static colorbar** with fixed `vmin/vmax` across all frames
  - Region clipping with optional padding

- **Performance & UX**
  - Parallel frame rendering (`--workers N`) + ffmpeg stitching
  - Persistent Cartopy cache (Natural Earth + tiles)
  - `.mplstyle` support for fonts/ticks
  - Headless-safe

---

## Install

### 0) Recommended environment
```bash
# conda/mamba is recommended for Cartopy/GEOS/PROJ
mamba create -n adcplot python=3.11
mamba activate adcplot

sudo apt-get update
sudo apt-get install -y libproj-dev proj-data proj-bin libgeos-dev ffmpeg
---

### Example command line flags
adcplot plot /path/to/fort.63.nc \
  --field zeta --tindex 0 \
  --basemap cartopy \
  --bounds "-81.8,30.6,-81.3,31.0" --clip --pad 0.02 \
  --vmin -1.5 --vmax 1.5 \
  --mpl-style ./mystyle.mplstyle \
  --out zeta_t0.png

adcplot animate /path/to/fort.63.nc \
  --field zeta --tstart 0 --tend 720 --fps 12 \
  --basemap tiles --tiles esri-imagery --tiles-zoom 12 \
  --bounds "-81.8,30.6,-81.3,31.0" --clip --pad 0.02 \
  --vmin -1.5 --vmax 1.5 \
  --colorbar-label "sea surface height (m)" \
  --title-tpl "{time} — Kings Bay Elevation (m)" \
  --workers 8 \
  --out zeta_cartopy.mp4

# magnitude from u,v
adcplot animate fort.64.nc \
  --magnitude-of u v --tstart 0 --tend 240 --fps 10 \
  --basemap cartopy \
  --bounds "-81.8,30.6,-81.3,31.0" --clip \
  --vmin 0 --vmax 2.5 \
  --out speed.mp4

# quiver (subsample with --stride)
adcplot animate fort.64.nc \
  --uv u v --vector-mode quiver --stride 8 --scale 60 \
  --basemap cartopy \
  --bounds "-81.8,30.6,-81.3,31.0" --clip \
  --fps 10 --tstart 0 --tend 240 --out currents.mp4

#Jupyter notebook example usage
from adcplot.registry import load_source, load_backend
from adcplot.plotter import Plotter

src = load_source("/path/to/fort.63.nc")
be = load_backend("mpl")
plotter = Plotter(src, be)

# Static map (t=0)
plotter.plot_static(
    field="zeta", tindex=0,
    bounds=(-81.8, 30.6, -81.3, 31.0), clip=True,
    vmin=-1.5, vmax=1.5,
    basemap="cartopy",
    out="zeta_t0.png",
)

# Animation (first 24 frames)
plotter.animate(
    field="zeta", tstart=0, tend=24, fps=12,
    basemap="tiles", tiles="esri-imagery", tiles_zoom=12,
    bounds=(-81.8, 30.6, -81.3, 31.0), clip=True, pad=0.02,
    vmin=-1.5, vmax=1.5,
    title_tpl="{time} — Kings Bay Elevation (m)",
    workers=4,
    out="zeta_day1.mp4",
)
