"""Plotting functions for SFINCS model data."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .utils import get_bounds_vector

__all__ = ["plot_forcing", "plot_basemap"]

geom_style = {
    "rivers": dict(linestyle="-", linewidth=1.0, color="darkblue"),
    "rivers_inflow": dict(linestyle=":", linewidth=1.0, color="darkblue"),
    "rivers_outflow": dict(linestyle=":", linewidth=1.0, color="darkgreen"),
    "msk2": dict(linestyle="-", linewidth=1.5, color="r"),
    "msk3": dict(linestyle="-", linewidth=1.5, color="m"),
    "thd": dict(linestyle="-", linewidth=1.0, color="k", annotate=False),
    "weir": dict(linestyle="--", linewidth=1.0, color="k", annotate=False),
    "bnd": dict(marker="^", markersize=75, c="w", edgecolor="k", annotate=True),
    "src": dict(marker=">", markersize=75, c="w", edgecolor="k", annotate=True),
    "obs": dict(marker="d", markersize=75, c="w", edgecolor="r", annotate=True),
    "crs": dict(linestyle="-", linewidth=1.5, color="deeppink", annotate=False),
    "region": dict(ls="--", linewidth=1, color="r"),
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
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    n = len(forcing.keys())
    kwargs0 = dict(sharex=True, figsize=(6, n * 3))
    kwargs0.update(**kwargs)
    fig, axes = plt.subplots(n, 1, **kwargs0)
    axes = [axes] if n == 1 else axes
    for i, name in enumerate(forcing):
        da = forcing[name].transpose("time", ...)
        longname = da.attrs.get("standard_name", "")
        unit = da.attrs.get("unit", "")
        prefix = ""
        if da.ndim == 3:
            if name.startswith("press"):
                da = da.min(dim=[da.raster.x_dim, da.raster.y_dim])
                prefix = "min "
            elif name.startswith("wind_u") or name.startswith("wind_v"):
                da = da.max(dim=[da.raster.x_dim, da.raster.y_dim])
                prefix = "max "
            else:
                da = da.mean(dim=[da.raster.x_dim, da.raster.y_dim])
                prefix = "mean "
        # convert to Single index dataframe (bar plots don't work with xarray)
        df = da.to_pandas()
        if isinstance(df.index, pd.MultiIndex):
            df = df.unstack(0)
        # convert dates a-priori as automatic conversion doesn't always work
        df.index = mdates.date2num(df.index)
        if name.startswith("precip"):
            df.plot(drawstyle="steps", ax=axes[i])
        elif (
            name.startswith("press")
            or name.startswith("wind10_u")
            or name.startswith("wind10_v")
        ):
            df.plot.line(ax=axes[i])
        elif name.startswith("wnd"):
            df.plot(ax=axes[i], kind="line", secondary_y="dir", legend=False)
            # set tick color for y-axis of variable 1
            axes[i].tick_params(axis="y", labelcolor="C0")
            axes[i].right_ax.set_ylabel("Wind direction [degrees]")
            # set tick color and label for y-axis of variable 2
            axes[i].right_ax.tick_params(axis="y", labelcolor="C1")

        else:
            df.plot.line(ax=axes[i]).legend(
                title="index",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                ncol=df.columns.size // 5 + 1,
                prop={"size": 8},
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
    ds: xr.Dataset,
    geoms: Dict,
    variable: str = "dep",
    shaded: bool = False,
    plot_bounds: bool = True,
    plot_region: bool = False,
    plot_geoms: bool = True,
    bmap: str = None,
    zoomlevel: int = "auto",
    figsize: Tuple[int] = None,
    geom_names: List[str] = None,
    geom_kwargs: Dict = {},
    legend_kwargs: Dict = {},
    bmap_kwargs: Dict = {},
    **kwargs,
):
    """Create basemap plot.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with model maps
    geoms : Dict of geopandas.GeoDataFrame
        Model geometries
    variable : str, optional
        Map of variable in ds to plot, by default 'dep'
    shaded : bool, optional
        Add shade to variable (only for variable = 'dep'), by default False
    plot_bounds : bool, optional
        Add waterlevel (msk=2) and open (msk=3) boundary conditions to plot.
    plot_region : bool, optional
        If True, plot region outline.
    plot_geoms : bool, optional
        If True, plot available geoms.
    bmap : str, optional
        background map souce name, by default None
        Default image tiles "sat", and "osm" are fetched from cartopy image tiles.
        If contextily is installed, xyzproviders tiles can be used as well.
    zoomlevel : int, optional
        zoomlevel, by default 'auto'
    figsize : Tuple[int], optional
        figure size, by default None
    geom_names : List[str], optional
        list of model geometries to plot, by default all model geometries
    geom_kwargs : Dict of Dict, optional
        Model geometry styling per geometry, passed to geopandas.GeoDataFrame.plot method.
        For instance: {'src': {'markersize': 30}}.
    legend_kwargs : Dict, optional
        Legend kwargs, passed to ax.legend method.

    Returns
    -------
    fig, axes
        Model fig and ax objects
    """
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    import matplotlib.pyplot as plt
    from matplotlib import colors, patheffects

    try:
        import contextily as cx

        has_cx = True
    except ImportError:
        has_cx = False

    # read crs and utm zone > convert to cartopy
    proj_crs = ds.raster.crs
    proj_str = proj_crs.name
    if proj_crs.is_projected and proj_crs.to_epsg() is not None:
        crs = ccrs.epsg(ds.raster.crs.to_epsg())
        unit = proj_crs.axis_info[0].unit_name
        unit = "m" if unit == "metre" else unit
        xlab, ylab = f"x [{unit}] - {proj_str}", f"y [{unit}]"
    elif proj_crs.is_geographic:
        crs = ccrs.PlateCarree()
        xlab, ylab = f"lon [deg] - {proj_str}", "lat [deg]"
    else:
        raise ValueError("Unsupported CRS")
    bounds = ds.raster.box.buffer(abs(ds.raster.res[0] * 1)).total_bounds
    extent = np.array(bounds)[[0, 2, 1, 3]]

    # create fig with geo-axis and set background
    if figsize is None:
        ratio = ds.raster.ycoords.size / (ds.raster.xcoords.size * 1.4)
        figsize = (6, 6 * ratio)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=crs)
    ax.set_extent(extent, crs=crs)
    if bmap is not None:
        if zoomlevel == "auto":  # auto zoomlevel
            c = 2 * np.pi * 6378137  # Earth circumference
            lat = np.array(ds.raster.transform_bounds(4326))[[1, 3]].mean()
            # max 4 x 4 tiles per image
            tile_size = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 4
            if proj_crs.is_geographic:
                tile_size = tile_size * 111111
            zoomlevel = int(np.log2(c * abs(np.cos(lat)) / tile_size))
            # sensible range is 9 (large metropolitan area) - 16 (street)
            zoomlevel = min(16, max(9, zoomlevel))
        #  short names for cartopy image tiles
        bmap = {"sat": "QuadtreeTiles", "osm": "OSM"}.get(bmap, bmap)
        if has_cx and bmap in list(cx.providers.flatten()):
            bmap_kwargs = dict(zoom=zoomlevel, **bmap_kwargs)
            cx.add_basemap(ax, crs=crs, source=bmap, **bmap_kwargs)
        elif hasattr(cimgt, bmap):
            bmap_img = getattr(cimgt, bmap)(**bmap_kwargs)
            ax.add_image(bmap_img, zoomlevel)
        else:
            err = f"Unknown background map: {bmap}"
            if not has_cx:
                err += " (note that contextily is not installed)"
            raise ValueError(err)

    # by default colorbar on lower right & legend upper right
    kwargs0 = {"cbar_kwargs": {"shrink": 0.5, "anchor": (0, 0)}}
    kwargs0.update(kwargs)
    # make nice cmap
    if "cmap" not in kwargs or "norm" not in kwargs:
        if variable == "dep" and "dep" in ds:
            vmin, vmax = ds["dep"].raster.mask_nodata().quantile([0.0, 0.98]).values
            vmin, vmax = int(kwargs.pop("vmin", vmin)), int(kwargs.pop("vmax", vmax))
            c_dem = plt.cm.terrain(np.linspace(0.25, 1, vmax))
            if vmin < 0:
                c_bat = plt.cm.terrain(np.linspace(0, 0.17, max(1, abs(vmin))))
                c_dem = np.vstack((c_bat, c_dem))
            cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap, norm = kwargs.pop("cmap", cmap), kwargs.pop("norm", norm)
            kwargs0.update(norm=norm, cmap=cmap)
        elif variable == "msk" and "msk" in ds:
            cmap = colors.LinearSegmentedColormap.from_list(
                "Set1", ["grey", "r", "m"], N=3
            )
            norm = colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3)
            kwargs0.update(norm=norm, cmap=cmap)
            kwargs0["cbar_kwargs"].update(ticks=[1, 2, 3])

    if variable in ds:
        da = ds[variable].raster.mask_nodata()
        if "msk" in ds and np.any(ds["msk"] > 0):
            da = da.where(ds["msk"] > 0)
        if da.raster.rotation != 0 and "xc" in da.coords and "yc" in da.coords:
            da.plot(transform=crs, x="xc", y="yc", ax=ax, zorder=1, **kwargs0)
        else:
            da.plot.imshow(transform=crs, ax=ax, zorder=1, **kwargs0)
        if shaded and variable == "dep" and da.raster.rotation == 0:
            ls = colors.LightSource(azdeg=315, altdeg=45)
            dx, dy = da.raster.res
            _rgb = ls.shade(
                da.fillna(0).values,
                norm=kwargs0["norm"],
                cmap=kwargs0["cmap"],
                blend_mode="soft",
                dx=dx,
                dy=dy,
                vert_exag=2,
            )
            rgb = xr.DataArray(
                dims=("y", "x", "rgb"), data=_rgb, coords=da.raster.coords
            )
            rgb = xr.where(np.isnan(da), np.nan, rgb)
            rgb.plot.imshow(transform=crs, ax=ax, zorder=1)

    # geometry plotting and annotate kwargs
    for k, d in geom_kwargs.items():
        geom_style[k].update(**d)
    ann_kwargs = dict(
        xytext=(3, 3),
        textcoords="offset points",
        zorder=4,
        path_effects=[
            patheffects.Stroke(linewidth=3, foreground="w"),
            patheffects.Normal(),
        ],
    )
    # plot mask boundaries
    if plot_bounds and "msk" not in ds:
        raise ValueError(
            "No 'msk' (sfincs.msk) found in ds required to plot the model bounds "
            "Set plot_bounds=False or add 'msk' to ds"
        )
    elif plot_bounds and (ds["msk"] >= 1).any():
        gdf_msk = get_bounds_vector(ds["msk"])
        gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]
        gdf_msk3 = gdf_msk[gdf_msk["value"] == 3]
        if gdf_msk2.index.size > 0:
            gdf_msk2.plot(ax=ax, zorder=3, label="waterlevel bnd", **geom_style["msk2"])
        if gdf_msk3.index.size > 0:
            gdf_msk3.plot(ax=ax, zorder=3, label="outflow bnd", **geom_style["msk3"])

    # plot static geoms
    if plot_geoms:
        geom_names = geom_names if isinstance(geom_names, list) else list(geoms.keys())
        for name in geom_names:
            gdf = geoms.get(name, None)
            if gdf is None or name in ["region", "bbox"]:
                continue
            # copy is important to keep annotate working if repeated
            kwargs = geom_style.get(name, {}).copy()
            annotate = kwargs.pop("annotate", False)
            gdf.plot(ax=ax, zorder=3, label=name, **kwargs)
            if annotate and np.all(gdf.geometry.type == "Point"):
                for label, row in gdf.iterrows():
                    x, y = row.geometry.x, row.geometry.y
                    ax.annotate(label, xy=(x, y), **ann_kwargs)

    if "region" in geoms and plot_region:
        geoms["region"].boundary.plot(
            ax=ax, zorder=2, label="region", **geom_style["region"]
        )

    # title, legend and labels
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    variable = "base" if variable is None else variable
    ax.set_title(f"SFINCS {variable} map")
    # NOTE without defined loc it takes forever to find a 'best' location
    # by default outside plot
    if geom_names or plot_bounds:
        legend_kwargs0 = dict(
            bbox_to_anchor=(1.05, 1),
            title="Legend",
            loc="upper left",
            frameon=True,
            prop=dict(size=8),
        )
        legend_kwargs0.update(**legend_kwargs)
        ax.legend(**legend_kwargs0)

    return fig, ax
