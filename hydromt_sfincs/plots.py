"""Plotting functions for SFINCS model data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
import matplotlib.pyplot as plt

# for animations
from matplotlib import animation
from IPython.display import HTML

from .utils import get_bounds_vector

logger = logging.getLogger(f"hydromt.{__name__}")

__all__ = ["plot_forcing", "plot_basemap", "plot_map_timesteps", "make_animation"]

geom_style = {
    "rivers": dict(linestyle="-", linewidth=1.0, color="darkblue"),
    "mask2": dict(linestyle="-", linewidth=1.5, color="r"),
    "mask3": dict(linestyle="-", linewidth=1.5, color="m"),
    "mask4": dict(linestyle="-", linewidth=1.5, color="g"),
    "mask5": dict(linestyle="-", linewidth=1.5, color="b"),
    "mask6": dict(linestyle="-", linewidth=1.5, color="orange"),
    "snapwave_bnd": dict(marker="^", markersize=60, c="cyan", edgecolor="k"),
    "thd": dict(linestyle="-", linewidth=1.0, color="k", annotate=False),
    "weir": dict(linestyle="--", linewidth=1.0, color="k", annotate=False),
    "bnd": dict(marker="^", markersize=75, c="w", edgecolor="k", annotate=True),
    "bdr": dict(color="g", linewidth=2, annotate=True, arrow=True),
    "src": dict(marker=">", markersize=75, c="w", edgecolor="k", annotate=True),
    "obs": dict(marker="d", markersize=75, c="w", edgecolor="r", annotate=True),
    "crs": dict(linestyle="-", linewidth=1.5, color="deeppink", annotate=False),
    "region": dict(ls="--", linewidth=1, color="r"),
}

# %% Original HydroMT-SFINCS setup_ functions:
# plot_basemap
# plot_forcing
# To add: geom_style?

# %% core HydroMT-SFINCS functions:
# _initialize
# plot_forcing
# plot_basemap


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
        elif name.startswith("wind"):
            df.plot(ax=axes[i], kind="line", secondary_y="direction", legend=False)
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


def plot_map_timesteps(
    da: Union[xr.DataArray, xu.UgridDataArray],
    col: str = "time",
    col_wrap: int = 3,
    time_step: int = 1,
    figsize: Tuple[int] = None,
    **kwargs,
):
    """Plot a spatial DataArray as facets along one dimension.

    Supports both regular xarray and unstructured xugrid DataArrays.
    ``time_step`` selects every Nth index along ``col``.
    Returns the figure and a 2D array of map axes.
    """
    import matplotlib.pyplot as plt

    if isinstance(da, xu.UgridDataArray):
        is_ugrid = True
    elif isinstance(da, xr.DataArray):
        is_ugrid = False
    else:
        raise TypeError("da must be an xarray or xugrid DataArray.")

    if col not in da.dims:
        raise ValueError(f"'{col}' must be a dimension of da.")
    if not isinstance(col_wrap, int) or col_wrap < 1:
        raise ValueError("col_wrap must be a positive integer.")
    if not isinstance(time_step, int) or time_step < 1:
        raise ValueError("time_step must be a positive integer.")

    da = da.isel({col: slice(None, None, time_step)})
    data = da.obj if is_ugrid else da
    n_frames = data.sizes[col]
    if n_frames == 0:
        raise ValueError(f"Dimension '{col}' is empty.")

    ncols = min(col_wrap, n_frames)
    nrows = (n_frames + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (5 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    plot_kwargs = kwargs.copy()
    add_colorbar = plot_kwargs.pop("add_colorbar", True)
    cbar_kwargs = plot_kwargs.pop("cbar_kwargs", {})
    if "norm" not in plot_kwargs:
        if plot_kwargs.get("vmin") is None:
            plot_kwargs["vmin"] = float(data.min(skipna=True).compute().item())
        if plot_kwargs.get("vmax") is None:
            plot_kwargs["vmax"] = float(data.max(skipna=True).compute().item())
    plot_kwargs["add_colorbar"] = False

    mappable = None
    for i, ax in enumerate(axes.flat[:n_frames]):
        frame = da.isel({col: i})
        if is_ugrid:
            mappable = frame.ugrid.plot(ax=ax, **plot_kwargs)
        else:
            mappable = frame.plot(ax=ax, **plot_kwargs)
        ax.set_title(f"{col} = {data[col].values[i]}")

    for ax in axes.flat[n_frames:]:
        ax.set_visible(False)

    if add_colorbar:
        fig.colorbar(mappable, ax=axes.flat[:n_frames].tolist(), **cbar_kwargs)

    return fig, axes


def _estimate_plot_limits(
    da: Union[xr.DataArray, xu.UgridDataArray],
) -> Tuple[float, float]:
    """Estimate robust color limits from the 0th and 98th percentiles."""
    vmin, vmax = da.quantile([0.0, 0.98], skipna=True).values
    return float(vmin), float(vmax)


def plot_basemap(
    ds: Union[xr.Dataset, xu.UgridDataset],
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
    grid_kwargs: Dict = {},
    legend_kwargs: Dict = {},
    bmap_kwargs: Dict = {},
    model: str = "sfincs",
    **kwargs,
):
    """Create basemap plot.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with model maps
    geoms : Dict of geopandas.GeoDataFrame
        Model geometries
    model : str, optional
        Model setup to plot, either ``"sfincs"`` (default) or ``"snapwave"``.
    variable : str, optional
        Map of variable in ds to plot, by default 'dep'
    shaded : bool, optional
        Add shade to variable (only for variable = 'dep'), by default False
    plot_bounds : bool, optional
        Plot boundary cells from the selected model mask.
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
    grid_kwargs : Dict, optional
        Styling options for grid plotting (e.g. color, linewidth) passed to matplotlib plot / ugrid.plot.line.
        Defaults: {"color": "black", "linewidth": 0.7}
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

    if model not in ("sfincs", "snapwave"):
        raise ValueError("model must be either 'sfincs' or 'snapwave'.")
    mask_name = "mask" if model == "sfincs" else "snapwave_mask"

    try:
        import contextily as cx

        has_cx = True
    except ImportError:
        has_cx = False

    # read crs and utm zone > convert to cartopy
    if isinstance(ds, xr.Dataset):
        proj_crs = ds.raster.crs
        bounds = ds.raster.box.buffer(abs(ds.raster.res[0] * 1)).total_bounds
        bbox = ds.raster.transform_bounds(4326)
        ratio = ds.raster.ycoords.size / (ds.raster.xcoords.size * 1.4)
    elif isinstance(ds, xu.UgridDataset):
        proj_crs = ds.grid.crs
        bounds = ds.ugrid.total_bounds
        bbox = ds.ugrid.to_crs(4326).ugrid.total_bounds
        ratio = (bounds[3] - bounds[1]) / ((bounds[2] - bounds[0]) * 1.4)
    proj_str = proj_crs.name
    extent = np.array(bounds)[[0, 2, 1, 3]]

    if proj_crs.is_projected and proj_crs.to_epsg() is not None:
        if "UTM" in proj_str:
            # Extract the UTM zone number
            utm_zone = proj_crs.utm_zone
            # Parse the zone number and hemisphere
            zone_number = int(utm_zone[:-1])  # Extract numeric part
            hemisphere = utm_zone[-1]  # Last character is 'N' or 'S'
            # Determine if it's in the southern hemisphere
            is_southern_hemisphere = hemisphere == "S"
            # Create the Cartopy UTM projection
            crs = ccrs.UTM(zone_number, southern_hemisphere=is_southern_hemisphere)
        else:
            crs = ccrs.epsg(proj_crs.to_epsg())

        # now check whether the model extent is within the extent of the crs
        bounds_crs = crs.boundary.bounds
        if (
            bounds[0] < bounds_crs[0]
            or bounds[1] < bounds_crs[1]
            or bounds[2] > bounds_crs[2]
            or bounds[3] > bounds_crs[3]
        ):
            logger.warning("Model domain exceeds the area of the CRS.")
            logger.warning("Consider reprojecting to EPSG:4326 for plotting.")
        unit = proj_crs.axis_info[0].unit_name
        unit = "m" if unit == "metre" else unit
        xlab, ylab = f"x [{unit}] - {proj_str}", f"y [{unit}]"
    elif proj_crs.is_geographic:
        crs = ccrs.PlateCarree()
        xlab, ylab = f"lon [deg] - {proj_str}", "lat [deg]"
    else:
        raise ValueError("Unsupported CRS")

    # create fig with geo-axis and set background
    if figsize is None:
        figsize = (6, 6 * ratio)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=crs)
    ax.set_extent(extent, crs=crs)
    bmap_has_attribution = False
    if bmap is not None:
        if zoomlevel == "auto":  # auto zoomlevel
            c = 2 * np.pi * 6378137  # Earth circumference
            lat = np.array(bbox)[[1, 3]].mean()
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
            source = cx.providers.flatten()[bmap]
            attribution = bmap_kwargs.get("attribution")
            if attribution is None:
                attribution = source.get("attribution")
            bmap_has_attribution = bool(attribution)
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
        depth_vars = ["dep", "z"]
        if variable in depth_vars and variable in ds:
            # auto-determine vmin and vmax
            vmin, vmax = _estimate_plot_limits(ds[variable].raster.mask_nodata())
            # overrule auto vmin and vmax with user input; keep as float for norm
            vmin = float(kwargs.pop("vmin", vmin))
            vmax = float(kwargs.pop("vmax", vmax))
            # make sure vmin and vmax are different after possible int truncation
            if vmax <= vmin:
                vmax = vmin + 1.0
            # build colormap: proportional sea/land sections based on data range
            n_land = max(2, round(max(0.0, vmax)))
            n_sea = max(1, round(-vmin)) if vmin < 0 else 0
            if vmax <= 0:
                # all terrain at or below sea level: use only bathymetric colors
                c_dem = plt.cm.terrain(np.linspace(0, 0.17, max(2, n_sea)))
            elif n_sea > 0:
                # terrain straddles sea level: combine sea and land colors
                c_bat = plt.cm.terrain(np.linspace(0, 0.17, n_sea))
                c_dem = plt.cm.terrain(np.linspace(0.25, 1, n_land))
                c_dem = np.vstack((c_bat, c_dem))
            else:
                # all terrain above sea level: use only land colors
                c_dem = plt.cm.terrain(np.linspace(0.25, 1, n_land))
            cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap, norm = kwargs.pop("cmap", cmap), kwargs.pop("norm", norm)
            kwargs0.update(norm=norm, cmap=cmap)
        elif variable in ("mask", "snapwave_mask") and variable in ds:
            mask_values = (
                [1, 2, 3] if variable == "snapwave_mask" else [1, 2, 3, 4, 5, 6]
            )
            cmap = colors.LinearSegmentedColormap.from_list(
                "Set1",
                ["grey", "r", "m", "g", "b", "orange"][: len(mask_values)],
                N=len(mask_values),
            )
            bounds = [value + 0.5 for value in range(len(mask_values) + 1)]
            norm = colors.BoundaryNorm(bounds, len(mask_values))
            kwargs0.update(norm=norm, cmap=cmap)
            kwargs0["cbar_kwargs"].update(ticks=mask_values)
    if variable in ds:
        da = ds[variable]
        if mask_name in ds and np.any(ds[mask_name] > 0):
            da = da.where(ds[mask_name] > 0)
        if isinstance(da, xr.DataArray):
            # mask_nodata converts it to an xr.DataArray, so performed inside if-else statement
            da = da.raster.mask_nodata()
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
        elif isinstance(da, xu.UgridDataArray):
            da.ugrid.plot(transform=crs, ax=ax, zorder=1, **kwargs0)
    elif variable == "grid":
        if isinstance(ds, xr.Dataset):
            grid_kwargs0 = {"color": "black", "linewidth": 0.7}
            grid_kwargs0.update(**grid_kwargs)
            if ds.raster.rotation != 0 and "xc" in ds.coords and "yc" in ds.coords:
                x, y = ds["xc"].values, ds["yc"].values
            else:
                x, y = np.meshgrid(ds["x"].values, ds["y"].values)

            ax.plot(
                x,
                y,
                zorder=1,
                transform=crs,
                **grid_kwargs0,
            )
        elif isinstance(ds, xu.UgridDataset):
            grid_kwargs0 = {"color": "black", "linewidth": 0.3}
            grid_kwargs0.update(**grid_kwargs)
            ds["level"].ugrid.plot.line(ax=ax, zorder=1, transform=crs, **grid_kwargs0)

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
    if plot_bounds and mask_name not in ds:
        logger.warning(
            "No '%s' found in ds required to plot the model bounds; "
            "skipping boundary plotting. Set plot_bounds=False or add the mask to ds.",
            mask_name,
        )
        plot_bounds = False
    if plot_bounds and (ds[mask_name] > 1).any():
        gdf_msk = get_bounds_vector(ds[mask_name])
        boundary_specs = {
            2: ("waves bnd", "mask2")
            if model == "snapwave"
            else ("waterlevel bnd", "mask2"),
            3: ("Neumann bnd", "mask3")
            if model == "snapwave"
            else ("outflow bnd", "mask3"),
        }
        if model == "sfincs":
            boundary_specs.update(
                {
                    4: ("wavemaker bnd", "mask4"),
                    5: ("downstream bnd", "mask5"),
                    6: ("Neumann bnd", "mask6"),
                }
            )
        for value, (label, style) in boundary_specs.items():
            gdf_boundary = gdf_msk[gdf_msk["value"] == value]
            if not gdf_boundary.empty:
                gdf_boundary.plot(ax=ax, zorder=3, label=label, **geom_style[style])

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
            arrow = kwargs.pop("arrow", False)
            if arrow and np.all(gdf.geometry.type == "LineString"):
                for _, row in gdf.iterrows():
                    coords = list(row.geometry.coords)
                    if len(coords) >= 2:
                        # arrow from 2nd point to 1st point
                        ax.annotate(
                            "",
                            xy=coords[0],  # arrow head
                            xytext=coords[1],  # arrow tail
                            arrowprops=dict(
                                arrowstyle="->",
                                color=kwargs.get("color", "k"),
                                linewidth=kwargs.get("linewidth", 1.5),
                            ),
                            zorder=4,
                            transform=crs,
                        )
            else:
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
    model_title = "SFINCS" if model == "sfincs" else "SnapWave"
    ax.set_title(f"{model_title} {variable} map")
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

    tile_name = "" if bmap is None else str(bmap).lower()
    is_satellite = any(
        keyword in tile_name for keyword in ("sat", "imagery", "aerial", "quadtree")
    )
    credit_x, credit_y, credit_va = (
        (0.01, 0.99, "top") if bmap_has_attribution else (0.01, 0.01, "bottom")
    )
    ax.text(
        credit_x,
        credit_y,
        "© Powered by HydroMT-SFINCS | Deltares",
        transform=ax.transAxes,
        ha="left",
        va=credit_va,
        fontsize=6,
        color="white" if is_satellite else "0.25",
        bbox=dict(
            facecolor="0.1" if is_satellite else "white",
            alpha=0.75,
            edgecolor="none",
            pad=1.5,
        ),
        zorder=10,
    )

    return fig, ax


def make_animation(
    data: Union[xr.DataArray, xr.Dataset, xu.UgridDataArray, xu.UgridDataset],
    variable: Optional[str] = None,
    geoms: Optional[Dict] = None,
    model: str = "sfincs",
    bmap: str = "sat",
    zoomlevel: Union[int, str] = "auto",
    figsize: Optional[Tuple[float, float]] = None,
    step: int = 1,
    fps: int = 4,
    dpi: int = 100,
    interval: int = 250,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    return_html: bool = True,
    **plot_kwargs,
) -> Optional[HTML]:
    """Create an inline animation of a time-dependent spatial variable.

    Regular-grid arrays are plotted using their native coordinates, including
    two-dimensional ``xc``/``yc`` coordinates for rotated grids. Quadtree
    arrays are plotted on their native unstructured mesh; they are not
    rasterized or resampled. Arguments are ordered by purpose: data selection,
    basemap options, animation styling, and output settings.

    Parameters
    ----------
    data : xr.DataArray, xr.Dataset, xu.UgridDataArray, or xu.UgridDataset
        Time-dependent spatial data. For a Dataset input, select the variable
        to animate with ``variable``.
    variable : str, optional
        Name of the variable to animate when ``data`` is a Dataset. Not needed
        when ``data`` is a DataArray.
    geoms : dict, optional
        Geometries to include in the basemap.
    model : {"sfincs", "snapwave"}, optional
        Model type passed to :func:`plot_basemap`, by default ``"sfincs"``.
    bmap : str, optional
        Background map name passed to :func:`plot_basemap`, by default ``"sat"``.
    zoomlevel : int or str, optional
        Background-map zoom level, by default ``"auto"``.
    figsize : tuple of float, optional
        Figure size in inches.
    step : int, optional
        Plot every ``step``-th timestep, by default 1.
    interval : int, optional
        Delay between animation frames in milliseconds, by default 250.
    title : str, optional
        Title prefix shown with each frame's timestamp. Defaults to the data's
        ``long_name`` attribute, its variable name, or ``"Data"``.
    fps : int, optional
        Frames per second when saving to a file, by default 4. Ignored when
        ``save_path`` is ``None``.
    dpi : int, optional
        Resolution used when saving to a file, by default 100. Ignored when
        ``save_path`` is ``None``.
    save_path : str or pathlib.Path, optional
        Optional output animation path. The file extension selects the
        Matplotlib writer. HTML output and MP4 files require the FFmpeg
        executable.
    return_html : bool, optional
        Return an HTML animation for inline display, by default True. Set to
        False to skip HTML generation; requires ``save_path``.
    **plot_kwargs
        Plot styling keywords passed to ``DataArray.plot`` or
        ``UgridDataArray.ugrid.plot``. By default, ``vmin`` and ``vmax`` are
        estimated from the full input using the 0th and 98th percentiles.
        Override either limit by passing ``vmin=...`` or ``vmax=...``. Other
        plot options such as ``cmap`` and ``alpha`` are also accepted. The
        ``cbar_kwargs`` keyword is passed to ``Figure.colorbar``, and
        ``add_colorbar=False`` disables the colorbar.

    Returns
    -------
    IPython.display.HTML or None
        HTML5 video for inline display, or ``None`` when ``return_html`` is
        False. HTML generation requires FFmpeg.

    Raises
    ------
    TypeError
        If ``data`` is not an xarray or xugrid DataArray or Dataset.
    ValueError
        If a Dataset is supplied without ``variable``, the input has no
        ``time`` dimension, ``step`` is not positive, or the number of spatial
        dimensions is unsupported, or HTML output is disabled without a
        ``save_path``.
    RuntimeError
        If HTML or MP4 output is requested but FFmpeg is unavailable.
    """
    if isinstance(data, (xr.Dataset, xu.UgridDataset)):
        if variable is None:
            raise ValueError("variable is required when data is a Dataset.")
        if variable not in data:
            raise KeyError(f"Variable {variable!r} is not present in data.")
        da = data[variable]
    elif isinstance(data, (xr.DataArray, xu.UgridDataArray)):
        da = data
    else:
        raise TypeError("data must be an xarray or xugrid DataArray or Dataset.")
    if not return_html and save_path is None:
        raise ValueError("save_path is required when return_html is False.")
    if "time" not in da.dims:
        raise ValueError("data must have a 'time' dimension.")
    if not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer.")
    if da.sizes["time"] == 0:
        raise ValueError("data has no timesteps to animate.")
    if save_path is not None:
        save_path = Path(save_path)
    needs_ffmpeg = return_html or (
        save_path is not None and save_path.suffix.lower() == ".mp4"
    )
    if needs_ffmpeg and not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "Returning inline HTML or saving an MP4 requires the FFmpeg "
            "executable. Install FFmpeg or disable HTML output when saving "
            "to a non-MP4 format."
        )

    is_ugrid = isinstance(da, xu.UgridDataArray)
    if is_ugrid:
        if da.isel(time=0).ndim != 1:
            raise ValueError("Quadtree animation data must have one spatial dimension.")
        x_name = y_name = None
    else:
        if da.isel(time=0).ndim != 2:
            raise ValueError(
                "Regular-grid animation data must have two spatial dimensions."
            )
        if "xc" in da.coords and "yc" in da.coords:
            x_name, y_name = "xc", "yc"
        else:
            x_name, y_name = da.raster.x_dim, da.raster.y_dim

    fig, ax = plot_basemap(
        ds=da.to_dataset(name=da.name or "data"),
        geoms=geoms or {},
        variable="",
        model=model,
        bmap=bmap,
        plot_bounds=False,
        zoomlevel=zoomlevel,
        figsize=figsize,
    )

    frame0 = da.isel(time=0)
    plot_options = plot_kwargs.copy()
    add_colorbar = plot_options.pop("add_colorbar", True)
    cbar_kwargs = plot_options.pop("cbar_kwargs", None)
    if "norm" not in plot_options and (
        plot_options.get("vmin") is None or plot_options.get("vmax") is None
    ):
        data_for_limits = da if is_ugrid else da.raster.mask_nodata()
        vmin, vmax = _estimate_plot_limits(data_for_limits)
        if plot_options.get("vmin") is None:
            plot_options["vmin"] = vmin
        if plot_options.get("vmax") is None:
            plot_options["vmax"] = vmax
    plot_options["ax"] = ax
    plot_options["add_colorbar"] = False
    projection = getattr(ax, "projection", None)
    if projection is not None:
        plot_options.setdefault("transform", projection)

    if is_ugrid:
        mappable = frame0.ugrid.plot(**plot_options)
    else:
        mappable = frame0.plot(
            x=x_name,
            y=y_name,
            **plot_options,
        )
    if add_colorbar:
        colorbar_options = {"shrink": 0.6, "anchor": (0, 0)}
        if cbar_kwargs is not None:
            colorbar_options.update(cbar_kwargs)
        fig.colorbar(mappable, ax=ax, **colorbar_options)

    title_prefix = title or da.attrs.get("long_name", da.name or "Data")

    def update_plot(i: int):
        frame = da.isel(time=i)
        timestamp = pd.Timestamp(da["time"].values[i]).strftime("%d-%B-%Y %H:%M:%S")
        ax.set_title(f"{title_prefix} {timestamp}")
        mappable.set_array(frame.values.ravel())
        return (mappable,)

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=np.arange(0, da.sizes["time"], step),
        interval=interval,
        blit=False,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), fps=fps, dpi=dpi)
    html = ani.to_html5_video() if return_html else None
    plt.close(fig)
    return HTML(html) if html is not None else None
