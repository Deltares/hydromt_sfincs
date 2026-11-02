import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu

from hydromt_sfincs import plots

import matplotlib.pyplot as plt


def test_plot_map_timesteps_ugrid():
    grid = xu.Ugrid2d(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        fill_value=-1,
        face_node_connectivity=np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        ),
    )
    da = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("time", grid.face_dimension),
        coords={"time": pd.date_range("2020-01-01", periods=3)},
        name="h",
    )
    uda = xu.UgridDataArray(da, grid=grid)

    fig, axes = plots.plot_map_timesteps(uda, col_wrap=3, time_step=2, vmax=4.0)
    try:
        assert axes.shape == (1, 2)
        assert len(fig.axes) == 3
        assert all(ax.collections for ax in axes.flat)
    finally:
        plt.close(fig)


def test_plot_map_timesteps_regular():
    da = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("2020-01-01", periods=2),
            "y": [0.0, 1.0],
            "x": [0.0, 1.0],
        },
        name="h",
    )

    fig, axes = plots.plot_map_timesteps(da, col_wrap=3, vmax=4.0)
    try:
        assert axes.shape == (1, 2)
        assert len(fig.axes) == 3
        assert all(ax.collections for ax in axes.flat)
    finally:
        plt.close(fig)


def _mock_html5_video(animation):
    animation._draw_was_started = True
    return "<video>animation</video>"


def _mock_basemap(monkeypatch):
    fig, ax = plt.subplots()
    monkeypatch.setattr(plots, "plot_basemap", lambda **kwargs: (fig, ax))
    monkeypatch.setattr(plots.animation.writers, "is_available", lambda writer: True)
    monkeypatch.setattr(
        plots.animation.FuncAnimation,
        "to_html5_video",
        _mock_html5_video,
    )
    return fig


def test_make_animation_regular_infers_rotated_coordinates(monkeypatch):
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    da = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("2020-01-01", periods=2),
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "xc": (("y", "x"), x + y * 0.1),
            "yc": (("y", "x"), y + x * 0.1),
        },
        name="temperature",
    )
    fig = _mock_basemap(monkeypatch)

    html = plots.make_animation(da)

    try:
        assert "animation" in html.data
        assert fig.axes[0].collections[0].get_clim() == pytest.approx((0.0, 6.86))
    finally:
        plt.close(fig)


def test_make_animation_selects_dataset_variable_and_forwards_basemap_options(
    monkeypatch,
):
    da = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("2020-01-01", periods=2),
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        },
        name="temperature",
    )
    ds = da.to_dataset()
    fig, ax = plt.subplots()
    basemap_kwargs = {}

    def mock_basemap(**kwargs):
        basemap_kwargs.update(kwargs)
        return fig, ax

    monkeypatch.setattr(plots, "plot_basemap", mock_basemap)
    monkeypatch.setattr(plots.animation.writers, "is_available", lambda writer: True)
    monkeypatch.setattr(
        plots.animation.FuncAnimation,
        "to_html5_video",
        _mock_html5_video,
    )
    geoms = {"obs": object()}

    html = plots.make_animation(
        ds,
        variable="temperature",
        geoms=geoms,
        model="snapwave",
        vmin=1.0,
        vmax=7.0,
        cmap="viridis",
        alpha=0.5,
        cbar_kwargs={"shrink": 0.4},
    )

    try:
        assert "animation" in html.data
        assert basemap_kwargs["model"] == "snapwave"
        assert basemap_kwargs["geoms"] is geoms
        assert (
            basemap_kwargs["ds"]["temperature"].drop_vars("spatial_ref").identical(da)
        )
        mappable = ax.collections[0]
        assert mappable.get_clim() == (1.0, 7.0)
        assert mappable.get_cmap().name == "viridis"
        assert mappable.get_alpha() == 0.5
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_make_animation_quadtree_uses_native_mesh(monkeypatch):
    grid = xu.Ugrid2d(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        fill_value=-1,
        face_node_connectivity=np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        ),
    )
    da = xr.DataArray(
        np.arange(8.0).reshape(2, 4),
        dims=("time", grid.face_dimension),
        coords={"time": pd.date_range("2020-01-01", periods=2)},
        name="zs",
    )
    uda = xu.UgridDataArray(da, grid=grid)
    fig = _mock_basemap(monkeypatch)

    html = plots.make_animation(uda)

    try:
        assert "animation" in html.data
    finally:
        plt.close(fig)


def test_make_animation_requires_positive_step():
    da = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    )

    with pytest.raises(ValueError, match="step must be a positive integer"):
        plots.make_animation(da, step=0)


def test_make_animation_requires_variable_for_dataset():
    ds = xr.Dataset(
        {"zs": (("time", "y", "x"), np.ones((1, 2, 2)))},
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    )

    with pytest.raises(ValueError, match="variable is required"):
        plots.make_animation(ds)


def test_make_animation_saves_without_generating_inline_html(monkeypatch, tmp_path):
    da = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    )
    saved = {}

    class MockAnimation:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path, fps, dpi):
            saved.update(path=path, fps=fps, dpi=dpi)

        def to_html5_video(self):
            raise AssertionError("HTML should not be generated.")

    monkeypatch.setattr(plots.animation, "FuncAnimation", MockAnimation)
    fig = _mock_basemap(monkeypatch)

    with pytest.raises(ValueError, match="save_path is required"):
        plots.make_animation(da, return_html=False)

    save_path = tmp_path / "animation.gif"
    result = plots.make_animation(
        da, save_path=save_path, return_html=False, fps=8, dpi=150
    )

    try:
        assert result is None
        assert saved == {"path": str(save_path), "fps": 8, "dpi": 150}
    finally:
        plt.close(fig)


def test_make_animation_reports_missing_ffmpeg(monkeypatch, tmp_path):
    da = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    )
    fig = _mock_basemap(monkeypatch)
    monkeypatch.setattr(plots.animation.writers, "is_available", lambda writer: False)

    with pytest.raises(RuntimeError, match="requires the FFmpeg executable"):
        plots.make_animation(da, save_path=tmp_path / "animation.mp4")

    with pytest.raises(RuntimeError, match="requires the FFmpeg executable"):
        plots.make_animation(da)

    plt.close(fig)
