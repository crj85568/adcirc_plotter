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

# smart_format
"""
Functions inside adcirc_plotting_combined.py
smart_format
generate_balanced_ticks
set_equal_visual_ticks
meters2deg
deg2meters
read14
read_adcirc_netcdf
plot_adcirc_mesh
preprocess_values
validate_bounds
setup_triangulation
init_figure
prepare_color
resolve_cmap
draw_tripcolor
add_mesh_lines
set_colorbar
configure_cbar_ticks
render_overlays
set_bounds
add_basemap_layer
plot_shapefile_overlay
add_state_overlays
plot_state
plot_geom_edges
finalize
dedupe_legend
normalize_bounds
animate_adcirc_zeta
is_netcdf_file
__main__
"""