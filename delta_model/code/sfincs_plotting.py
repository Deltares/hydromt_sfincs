"""
HydroMT-SFINCS utilities functions for reading and writing SFINCS specific input and output files,
as well as some common data conversions.
"""

from pathlib import Path
from typing import Tuple, Union

import bokeh as bk
import holoviews as hv
import holoviews.operation.datashader as hd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import xarray as xr
from hydromt import open_raster
from hydromt_sfincs.plots import plot_basemap
from IPython.display import HTML
from matplotlib import animation


def make_interactive_plot(
        da_fn: Union[str, Path], 
        bmap: str = 'OpenStreetMap', 
        title: str = 'Flood map',
        vmin: float = None,
        vmax: float = None
        ):
    """Create an interactive flood map using HoloViews and Panel.

    Parameters
    ----------
    da_fn : str, Path
        Path to the raster dataset to be visualized.
    bmap : str
        Name of the background map to be used.
        Options are 'OpenStreetMap' (default), 'Satellite', 'CartoDark', 'CartoLight'. 
    title : str
        Title of the figure.
    vmin : float, optional
        Minimum color limit for the colormap.
    vmax : float, optional
        Maximum color limit for the colormap.

    """
    
    hv.extension('bokeh', logo=False)

    # Define a custom colormap
    colormap_terrain_cut = [matplotlib.colors.rgb2hex(c) for c in plt.cm.terrain(np.linspace(0.25, 1, 192))]

    # Define dimensions and data
    key_dimensions = ['x', 'y']
    value_dimension = "elevation"
    image_width, image_height = 600, 600
    map_width, map_height = 900, 600

    # Read and preprocess the raster data
    dataarray = open_raster(da_fn).raster.reproject('EPSG:3857')
    dataarray = dataarray.raster.mask_nodata()

    # Define a custom HoverTool formatter code
    formatter_code = """
      var digits = 4;
      var projections = Bokeh.require("core/util/projections");
      var x = special_vars.x; var y = special_vars.y;
      var coords = projections.wgs84_mercator.invert(x, y);
      return "" + (Math.round(coords[%d] * 10**digits) / 10**digits).toFixed(digits)+ "";
    """
    formatter_code_x, formatter_code_y = formatter_code % 0, formatter_code % 1
    custom_tooltips = [('Lon', '@x{custom}'), ('Lat', '@y{custom}'), ('Flood depth', '@image{0.00} m')]
    custom_formatters = {
        '@x': bk.models.CustomJSHover(code=formatter_code_x),
        '@y': bk.models.CustomJSHover(code=formatter_code_y)
    }
    custom_hover = bk.models.HoverTool(tooltips=custom_tooltips, formatters=custom_formatters)

    hv.opts.defaults(
        hv.opts.Image(height=image_height,
                      width=image_width,
                      colorbar=True, 
                      tools=[custom_hover], 
                      active_tools=['wheel_zoom'],
                      clipping_colors={'NaN': '#00000000'}),
        hv.opts.Tiles(active_tools=['wheel_zoom'])
    )

    # Create an HoloViews Dataset
    dataset = hv.Dataset(dataarray[0], vdims=value_dimension, kdims=key_dimensions)
    image = hv.Image(dataset)
    min_val = float(dataset.data[value_dimension].min())
    max_val = float(dataset.data[value_dimension].max())
    minmaxes = (vmin if vmin is not None else min_val, vmax if vmax is not None else max_val)

    pn_colormap = pn.widgets.Select(options=['blues','viridis', 'summer', 'topography'], value='blues', name='Colormap')
    pn_opacity = pn.widgets.FloatSlider(name='Opacity', value=0.8, start=0, end=1, step=0.1)

    tile_sources = {
        'OpenStreetMap': hv.element.tiles.OSM,
        'Satellite': hv.element.tiles.EsriImagery,
        'CartoDark': hv.element.tiles.CartoDark,
        'CartoLight': hv.element.tiles.CartoLight,
        # Add more tile sources as needed
    }
    
    if bmap not in tile_sources.keys():
        raise ValueError(f'Background map {bmap} not available. Please select one of {tile_sources.keys()}.')

    # Create the selected background tile source
    tiles = tile_sources[bmap]().options(width=map_width, height=map_height)

    @pn.depends(
        pn_colormap_value=pn_colormap.param.value,
        pn_opacity_value=pn_opacity.param.value
    )
    def load_map(pn_colormap_value, pn_opacity_value):
        if pn_colormap_value == 'topography':
            used_colormap = colormap_terrain_cut
        else:
            used_colormap = pn_colormap_value

        image.opts(cmap=used_colormap, alpha=pn_opacity_value)
        return image

    dynmap = hd.regrid(hv.DynamicMap(load_map)).opts(clim=minmaxes)

    combined = tiles * dynmap

    return pn.Column(
        pn.WidgetBox(f'## {title}',  # Use the provided figure title
                     pn.Row(
                         pn.Row(pn_colormap, width=int(map_width/2)),
                         pn.Row(pn_opacity, width=int(map_width/2)),
                     ),
        ),
        combined
    )

def make_animation(
    da_zs: xr.DataArray, 
    geoms: dict, 
    bmap: str = "sat",
    zoomlevel: int = "auto",
    plot_bounds: bool = False,
    figsize: Tuple[int] = None,
    step=1, 
    ):

    def update_plot(i, da_zs, cax_h):
        da_zs = da_zs.isel(time=i)
        t = da_zs.time.dt.strftime("%d-%B-%Y %H:%M:%S").item()
        ax.set_title(f"SFINCS water depth {t}")
        cax_h.set_array(da_zs.values.ravel())

    fig, ax = plot_basemap(
        ds = da_zs.to_dataset(),
        geoms = geoms,
        variable="",
        plot_bounds=plot_bounds,
        bmap=bmap,
        zoomlevel=zoomlevel,
        figsize=figsize,
    )

    cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}

    cax_h = da_zs.isel(time=0).plot(x="x", y="y", ax=ax, vmin=0, vmax=3, 
        cmap=plt.cm.Blues, alpha=0.75, cbar_kwargs=cbar_kwargs)
    plt.close()  # to prevent double plot

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=np.arange(0, da_zs.time.size, step),
        interval=250,  # ms between frames
        fargs=(da_zs, cax_h,),
    )

    # to show in notebook:
    return HTML(ani.to_html5_video())