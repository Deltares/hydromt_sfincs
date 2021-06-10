import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, Tuple, List

__all__ = ["plot_forcing", "plot_basemap"]

geom_style = {
    "rivers": dict(linestyle="--", linewidth=1.0, color="b"),
    "rivers_out": dict(linestyle="--", linewidth=1.0, color="r"),
    "bnd": dict(marker="^", markersize=75, c="w", edgecolor="k", annotate=True),
    "src": dict(marker=">", markersize=75, c="w", edgecolor="k", annotate=True),
    "obs": dict(marker="d", markersize=75, c="w", edgecolor="r", annotate=True),
}


def plot_forcing(forcing: Dict, **kwargs):
    """Plot model timeseries forcing.

    For distributed forcing a spatial avarage is plotted.

    Parameters
    ----------
    forcing : Dict of xr.DataArray
        Model forcing

    Returns
    -------
    fig, axes
        Model fig and ax objects
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    n = len(forcing.keys())
    kwargs0 = dict(sharex=True, figsize=(6, n * 3))
    kwargs0.update(**kwargs)
    fig, axes = plt.subplots(n, 1, **kwargs0)
    axes = [axes] if n == 1 else axes
    for i, name in enumerate(forcing):
        da = forcing[name]
        prefix = ""
        if da.ndim == 3:
            da = da.mean(dim=[da.raster.x_dim, da.raster.y_dim])
            prefix = "mean "
        # convert to Single index dataframe (bar plots don't work with xarray)
        df = da.squeeze().to_series()
        if isinstance(df.index, pd.MultiIndex):
            df = df.unstack(0)
        # convert dates a-priori as automatic conversion doesn't always work
        df.index = mdates.date2num(df.index)
        longname = da.attrs.get("standard_name", "")
        unit = da.attrs.get("unit", "")
        if longname == "precipitation":
            axes[i].bar(df.index, df.values, facecolor="darkblue")
        else:
            df.plot.line(ax=axes[i]).legend(
                title="index",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                ncol=2,
            )
        axes[i].set_ylabel(f"{prefix}{longname}\n[{unit}]")
        axes[i].set_title(f"SFINCS {longname} forcing ({name})")

    # use a concise date formatter for format date axis ticks
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    return fig, axes


def plot_basemap(
    staticmaps: xr.Dataset,
    staticgeoms: Dict,
    variable: str = "dep",
    shaded: bool = True,
    bmap: str = "sat",
    zoomlevel: int = 11,
    figsize: Tuple[int] = None,
    geoms: List[str] = ["rivers", "src", "bnd", "obs"],
    geom_kwargs: Dict = {},
    legend_kwargs: Dict = {},
    **kwargs,
):
    """Create basemap plot.

    Parameters
    ----------
    staticmaps : xr.Dataset
        Dataset with model maps
    staticgeoms : Dict of geopandas.GeoDataFrame
        Model geometries
    variable : str, optional
        Map name to plot, by default 'dep'
    shaded : bool, optional
        Add shaded (only if variable is True), by default True
    bmap : {'sat', ''}
        background map, by default "sat"
    zoomlevel : int, optional
        zoomlevel, by default 11
    figsize : Tuple[int], optional
        figure size, by default None
    geoms : List[str], optional
        list of model geometries to plot, by default ["rivers", "src", "bnd", "obs"]
    geom_kwargs : Dict, optional
        Model geometry styling, passed to geopands.GeoDataFrame.plot method
    legend_kwargs : Dict, optional
        Legend kwargs, passed to ax.legend method.

    Returns
    -------
    fig, axes
        Model fig and ax objects
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors, patheffects
    import cartopy.io.img_tiles as cimgt
    import cartopy.crs as ccrs

    # read crs and utm zone > convert to cartopy
    wkt = staticmaps.raster.crs.to_wkt()
    if "UTM zone " not in wkt:
        raise ValueError("Model CRS UTM zone not found.")
    utm_zone = staticmaps.raster.crs.to_wkt().split("UTM zone ")[1][:3]
    utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
    extent = np.array(staticmaps.raster.box.buffer(2e3).total_bounds)[[0, 2, 1, 3]]

    # create fig with geo-axis and set background
    if figsize is None:
        ratio = staticmaps.raster.ycoords.size / (staticmaps.raster.xcoords.size * 1.2)
        figsize = (10, 10 * ratio)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=utm)
    ax.set_extent(extent, crs=utm)
    if bmap == "sat":
        ax.add_image(cimgt.QuadtreeTiles(), zoomlevel)
    elif bmap == "osm":
        ax.add_image(cimgt.OSM(), zoomlevel)

    # make nice cmap
    if "cmap" not in kwargs or "norm" not in kwargs:
        if variable == "dep":
            vmin, vmax = da = (
                staticmaps[variable].raster.mask_nodata().quantile([0.0, 0.98])
            )
            vmin, vmax = kwargs.pop("vmin", vmin), kwargs.pop("vmax", vmax)
            c_bat = plt.cm.terrain(np.linspace(0, 0.17, 256))
            c_dem = plt.cm.terrain(np.linspace(0.25, 1, 256))
            if vmin < 0:
                c_all = np.vstack((c_bat, c_dem))
                cmap = colors.LinearSegmentedColormap.from_list("bat_dem", c_all)
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            else:
                cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap, norm = kwargs.pop("cmap", cmap), kwargs.pop("norm", norm)
            kwargs.update(norm=norm, cmap=cmap)
    if variable in staticmaps:
        da = staticmaps[variable].raster.mask_nodata()
        da.plot(transform=utm, ax=ax, zorder=1, **kwargs)
        if shaded and variable == "dep":
            ls = colors.LightSource(azdeg=315, altdeg=45)
            dx, dy = da.raster.res
            _rgb = ls.shade(
                da.fillna(0).values,
                norm=kwargs["norm"],
                cmap=kwargs["cmap"],
                blend_mode="soft",
                dx=dx,
                dy=dy,
                vert_exag=50,
            )
            rgb = xr.DataArray(
                dims=("y", "x", "rgb"), data=_rgb, coords=da.raster.coords
            )
            rgb = xr.where(np.isnan(da), np.nan, rgb)
            rgb.plot.imshow(transform=utm, ax=ax, zorder=1)

    # TODO add vectorized boundary (points) from msk==2 and msk==3.

    # add geoms
    geom_style.update(geom_kwargs)
    ann_kwargs = dict(
        xytext=(3, 3),
        textcoords="offset points",
        zorder=4,
        path_effects=[
            patheffects.Stroke(linewidth=3, foreground="w"),
            patheffects.Normal(),
        ],
    )
    if staticgeoms:
        for name in geoms:
            gdf = staticgeoms.get(name, None)
            if gdf is None:
                continue
            annotate = geom_style[name].pop("annotate", False)
            gdf.plot(ax=ax, zorder=3, **geom_style[name], label=name)
            if annotate:
                for label, row in gdf.iterrows():
                    x, y = row.geometry.x, row.geometry.y
                    ax.annotate(label, xy=(x, y), **ann_kwargs)
        if "region" in staticgeoms:
            staticgeoms["region"].boundary.plot(
                ax=ax, ls="-", lw=0.5, color="k", zorder=2
            )

    # title, legend and labels
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel(f"y coordinate UTM zone {utm_zone} [m]")
    ax.set_xlabel(f"x coordinate UTM zone {utm_zone} [m]")
    variable = "base" if variable is None else variable
    ax.set_title(f"SFINCS {variable} map")
    # NOTE without defined loc it takes forever to find a 'best' location
    legend_kwargs0 = dict(
        title="Legend",
        loc="lower right",
        frameon=True,
        framealpha=0.7,
    )
    legend_kwargs0.update(**legend_kwargs)
    ax.legend(**legend_kwargs0)

    return fig, ax
