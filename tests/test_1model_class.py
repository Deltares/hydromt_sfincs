"""Test sfincs model class against hydromt.models.model_api"""

import os
from os.path import isfile, join

import numpy as np
import math
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, Point
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from hydromt.readers import read_workflow_yaml

# from hydromt.log import setuplog

from hydromt_sfincs.sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR

_cases = {
    "test1": {
        "ini": "sfincs_test.yml",
        "example": "sfincs_test",
    },
    "test2": {
        "example": "sfincs_test_quadtree",
    },
}


@pytest.mark.parametrize("case", list(_cases.keys())[:1])
def test_model_build(tmpdir, case):
    # compare results with model from examples folder
    root = str(tmpdir.join(case))
    root0 = TESTMODELDIR

    # FIXME change directory into testdata folder to find data catalog ... is this logical?
    os.chdir(TESTDATADIR)

    # Build model
    config = join(TESTDATADIR, _cases[case]["ini"])
    modeltype, kwargs, steps = read_workflow_yaml(config, modeltype="sfincs")

    # logger = setuplog(path=join(root, "hydromt.log"), log_level=10)
    mod1 = SfincsModel(root=root, mode="w", **kwargs)
    # convert steps to list of dicts
    mod1.build(steps=steps)
    # Check if model is api compliant
    # non_compliant_list = mod1.test_model_api()
    # assert len(non_compliant_list) == 0

    # read and compare with model from examples folder
    mod0 = SfincsModel(root=root0, mode="r")
    mod0.read()
    mod1 = SfincsModel(root=root, mode="r")
    mod1.read()

    # compare config
    d0 = mod0.config.data.model_dump()
    d1 = mod1.config.data.model_dump()

    def equal(a, b, tol=1e-6):
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
        return a == b

    # ignore some keys that we know are different
    ignore_keys = {"bzsfile", "bndfile", "bcafile", "netbndbzsbzifile", "wvmfile"}
    diff = {
        k: (d0.get(k), d1.get(k))
        for k in d0.keys() | d1.keys()
        if k not in ignore_keys and not equal(d0.get(k), d1.get(k))
    }
    assert not diff, f"Differences:\n{diff}"

    # check grid
    invalid_maps = []
    if len(mod0.grid.data) > 0:
        assert np.all(mod0.crs == mod1.crs), "map crs"
        mask = (mod0.grid.data["mask"] > 0).values  # compare only active cells
        mask1 = (mod1.grid.data["mask"] > 0).values
        assert np.allclose(mask, mask1), "mask mismatch"
        for name in mod0.grid.data.raster.vars:
            if name == "mask":
                continue
            map0 = mod0.grid.data[name].values
            map1 = mod1.grid.data[name].values
            if not np.allclose(map0[mask], map1[mask]):
                invalid_maps.append(name)
    invalid_map_str = ", ".join(invalid_maps)
    assert len(invalid_maps) == 0, f"invalid maps: {invalid_map_str}"
    # check geometries
    geom_components = [
        "observation_points",
        "cross_sections",
        "thin_dams",
        "weirs",
        "drainage_structures",
    ]
    invalid_geoms = []
    for name in geom_components:
        if mod0.components[name].data.empty:
            continue
        try:
            assert_geodataframe_equal(
                mod0.components[name].data,
                mod1.components[name].data,
                check_less_precise=True,  # allow for rounding errors in geoms
                check_like=True,  # order may be different
                check_geom_type=True,  # geometry types should be the same
                normalize=True,  # normalize geometry
            )
        except AssertionError:  # re-raise error with geom name
            invalid_geoms.append(name)
    assert len(invalid_geoms) == 0, f"invalid geoms: {invalid_geoms}"
    # check forcing conditions
    forcing_components = [
        ("water_level", "bzs"),
        ("discharge_points", "dis"),
        ("wind", "wind"),
        ("precipitation", "precip_2d"),
        # ("pressure", "prs"),
    ]
    invalid_forcing = []
    for comp, name in forcing_components:
        data = mod0.components[comp].data[name]
        if isinstance(data, gpd.GeoDataFrame) and not data.empty:
            try:
                assert_geodataframe_equal(
                    mod0.components[comp].data,
                    mod1.components[comp].data,
                    check_less_precise=True,  # allow for rounding errors in geoms
                    check_like=True,  # order may be different
                    check_geom_type=True,  # geometry types should be the same
                    normalize=True,  # normalize geometry
                )
            except AssertionError:  # re-raise error with geom name
                invalid_forcing.append(comp)
        elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
            # Only compare if mod0.forcing[name] is not empty
            data0 = mod0.components[comp].data[name]
            data1 = mod1.components[comp].data[name]
            if (isinstance(data0, xr.DataArray) and data0.size > 0) or (
                isinstance(data0, xr.Dataset) and len(data0.data_vars) > 0
            ):
                try:
                    assert np.allclose(data0.values, data1.values, atol=1.0e-2)
                except AssertionError:  # re-raise error with forcing name
                    invalid_forcing.append(comp)
        assert len(invalid_forcing) == 0, f"invalid forcing: {invalid_forcing}"


def test_infiltration(model):
    # set constant infiltration
    qinf = xr.where(model.grid.data["dep"] < -0.5, -9999, 0.1)
    qinf.raster.set_nodata(-9999.0)
    qinf.raster.set_crs(model.crs)
    model.infiltration.create_constant(qinf, reproj_method="nearest")
    assert model.config.get("qinf") is None  # qinf removed from config
    assert model.config.get("qinffile") is not None  # qinf file set
    assert "qinf" in model.grid.data

    # set cn infiltration
    cn = xr.where(model.grid.data["dep"] < -0.5, 0, 50)
    cn.raster.set_nodata(-1)
    cn.raster.set_crs(model.crs)
    model.infiltration.create_cn(cn, reproj_method="nearest")
    assert model.config.get("scsfile") is not None  # scs file set
    assert "scs" in model.grid.data
    assert (model.grid.data["scs"].where(model.grid.mask > 0)).min() == 10
    assert model.config.get("qinffile") is None  # qinf file  reset

    # set cn infiltration with recovery
    lulc = xr.where(model.grid.data["dep"] < -0.5, 70, 30)
    lulc.raster.set_crs(model.crs)
    hsg = xr.where(model.grid.data["dep"] < 2, 1, 3)
    hsg.raster.set_crs(model.crs)
    ksat = xr.where(model.grid.data["dep"] < 1, 0.01, 0.2)
    ksat.raster.set_crs(model.crs)
    # create pandas reclass table for lulc and hsg to cn
    reclass_table = pd.DataFrame([[0, 35], [0, 56]], index=[70, 30], columns=[1, 3])
    effective = 0.5
    model.infiltration.create_cn_with_recovery(
        lulc=lulc, hsg=hsg, ksat=ksat, reclass_table=reclass_table, effective=effective
    )

    # Check if variables are there
    assert "smax" in model.grid.data
    assert "seff" in model.grid.data
    assert "ks" in model.grid.data
    assert model.config.get("scsfile") is None  # scs file reset

    # Write model
    model.grid.write()
    model.config.write()

    # read and check if identical
    mod1 = SfincsModel(root=model.root.path, mode="r")
    mod1.config.read()
    mod1.grid.read()

    # assure the sum of smax is close to earlier calculated value
    assert np.isclose(mod1.grid.data["smax"].where(mod1.grid.mask > 0).sum(), 32.929287)
    assert np.isclose(
        mod1.grid.data["seff"].where(mod1.grid.mask > 0).sum(), 32.929287 * effective
    )
    assert np.isclose(mod1.grid.data["ks"].where(mod1.grid.mask > 0).sum(), 331.27203)


def test_initial_conditions(model):
    # set spatially varying initial waterlevel
    ini = xr.where(model.grid.data["dep"] < -0.5, np.nan, 0.5)
    # ini.raster.set_nodata(-9999.0)
    ini.raster.set_crs(model.crs)
    model.initial_conditions.create(ini, reproj_method="nearest")
    assert model.config.get("zsini") is None  # zsini removed from config
    assert model.config.get("inifile") is not None  # inifile set
    assert "ini" in model.grid.data

    # Write model
    model.grid.write()
    model.config.write()

    # read and check if identical
    mod1 = SfincsModel(root=model.root.path, mode="r")
    mod1.config.read()
    mod1.grid.read()

    # assure the sum of ini is close to earlier calculated value
    assert np.isclose(
        mod1.grid.data["ini"]
        .where((mod1.grid.mask > 0) & (mod1.grid.data["ini"].values > 0))
        .sum(),
        890.5,
    )


def test_initial_conditions_from_polygon(model):
    # set spatially varying initial waterlevel
    region = model.data_catalog.get_geodataframe(
        join(TESTDATADIR, "region.geojson"),
    )
    region["ini"] = 0.5

    model.initial_conditions.create_from_polygon(region, reset_ini=True)

    # check if values are correctly set
    assert model.config.get("zsini") is None  # zsini removed from config
    assert model.config.get("inifile") is not None  # inifile set
    assert "ini" in model.grid.data

    # Write model
    model.grid.write()
    model.config.write()

    # read and check if identical
    mod1 = SfincsModel(root=model.root.path, mode="r")
    mod1.config.read()
    mod1.grid.read()

    # assure the sum of ini is close to earlier calculated value
    assert np.isclose(
        mod1.grid.data["ini"]
        .where((mod1.grid.mask > 0) & (mod1.grid.data["ini"].values > 0))
        .sum(),
        1139.5,
    )


def test_subgrid_io(model_config, tmp_dir):
    # test the backward compatibility of reading/writing subgrid

    # read-in the current subgrid (netcdf format)
    model_config.config.read()
    model_config.grid.read()
    model_config.subgrid.read()

    # check version and new parameter
    assert model_config.subgrid.version == 1
    # u and v paramters should be separated internally
    assert "u_pwet" in model_config.subgrid.data
    assert "uv_pwet" not in model_config.subgrid.data

    # also read-in the "real" netcdf file wihtout any hydromt interpretation
    sbg0 = xr.load_dataset(model_config.root.path / "sfincs_subgrid.nc")

    # write the subgrid (new format)
    tmp_root = tmp_dir / "subgrid_io_test"
    model_config.root.set(tmp_root, mode="w")
    model_config.write()
    assert isfile(join(model_config.root.path / "sfincs_subgrid.nc"))

    # read back-in
    mod1 = SfincsModel(root=tmp_root, mode="r")
    mod1.read()
    # Check if variables are the same
    assert (
        model_config.subgrid.data.variables.keys() == mod1.subgrid.data.variables.keys()
    )

    # Check if values are almost equal
    for var_name in model_config.subgrid.data.variables:
        assert (
            np.sum(model_config.subgrid.data[var_name] - mod1.subgrid.data[var_name])
            == 0.0
        )

    # now read again the raw-netcdf file without any hydromt interpretation
    sbg1 = xr.load_dataset(mod1.root.path / "sfincs_subgrid.nc")

    # Check if values are almost equal
    for var_name in sbg0.variables:
        assert np.sum(sbg0[var_name] - sbg1[var_name]) == 0.0

    # copy old sbgfile to new location
    sbgfile = join(TESTDATADIR, "sfincs_test", "sfincs.sbg")

    # change the subgrid to the old format (binary format)
    mod1.config.set("sbgfile", sbgfile)
    mod1.subgrid.read()

    # NOTE values are not the same as in the new format due to some changes in #225 and #247
    # only check version and new parameter
    assert mod1.subgrid.version == 0
    assert "u_pwet" not in mod1.subgrid.data
    assert "uv_pwet" not in mod1.subgrid.data


def test_subgrid_rivers(model):
    gdf_riv = model.data_catalog.get_geodataframe(
        "hydro_rivers_lin", geom=model.region, buffer=1e3
    )

    # create dummy depths for the river based on the width
    rivdph = gdf_riv["rivwth"].values / 100
    gdf_riv["rivdph"] = rivdph

    # set the depth of the river with "COMID": 21002062 to nan
    gdf_riv.loc[gdf_riv["COMID"] == 21002062, "rivdph"] = np.nan

    sbg_org = model.subgrid.data.copy()

    model.subgrid.create(
        elevation_list=[
            {"elevation": "merit_hydro", "zmin": 0.001},
            {"elevation": "gebco"},
        ],
        roughness_list=[
            {
                "lulc": "vito_2015",
                "reclass_table": join(TESTDATADIR, "local_data", "vito_mapping.csv"),
            }
        ],
        river_list=[
            {
                "centerlines": gdf_riv,
                "rivdph": 1,
                "rivwth": 100,
                "manning": 0.035,
            }
        ],
        write_dep_tif=True,
        write_man_tif=True,
        nr_subgrid_pixels=6,
        nr_levels=8,
        nrmax=250,  # multiple tiles
    )

    assert isfile(model.root.path / "subgrid" / "dep_subgrid.tif")
    assert isfile(model.root.path / "subgrid" / "manning_subgrid.tif")

    assert np.isclose(
        np.sum(sbg_org["z_zmin"] - model.subgrid.data["z_zmin"]), 124.13107
    )


def test_structs(model_config, tmp_dir):
    # read
    model_config.grid.read()
    model_config.thin_dams.read()
    assert not model_config.thin_dams.data.empty
    nr_thin_dams = len(model_config.thin_dams.data.index)
    # write thd file only
    tmp_root = tmp_dir / "struct_test"
    model_config.root.set(tmp_root, mode="w+")
    model_config.thin_dams.write()
    assert isfile(join(model_config.root.path, "sfincs.thd"))
    fn_thd_gis = join(model_config.root.path, "gis", "thd.geojson")
    assert isfile(fn_thd_gis)
    # add second thd file
    model_config.thin_dams.create(fn_thd_gis, merge=True)
    assert len(model_config.thin_dams.data.index) == nr_thin_dams * 2
    # setup weir file from thd.geojson using dz option
    with pytest.raises(ValueError, match="Weir structure requires z"):
        model_config.weirs.create(fn_thd_gis)
    model_config.weirs.create(fn_thd_gis, dz=2)
    assert not model_config.weirs.data.empty
    assert model_config.config.get("weirfile") is not None
    model_config.weirs.write()
    model_config.thin_dams.write()
    assert isfile(join(model_config.root.path, "sfincs.weir"))
    fn_weir_gis = join(model_config.root.path, "gis", "weir.geojson")
    assert isfile(fn_weir_gis)
    # test with buffer
    model_config.weirs.create(fn_thd_gis, buffer=5, dep="dep", merge=False)
    assert len(model_config.weirs.data.index) == 2


def test_drainage_structures(model_config, tmp_dir):
    model_config.drainage_structures.read()
    assert not model_config.drainage_structures.data.empty
    nr_drainage_structures = len(model_config.drainage_structures.data.index)
    # write drn file only
    tmp_root = tmp_dir / "drainage_struct_test"
    model_config.root.set(tmp_root, mode="w+")
    model_config.drainage_structures.write()
    assert isfile(model_config.root.path / "sfincs.drn")
    fn_drn_gis = join(model_config.root.path, "gis", "drn.geojson")
    assert isfile(fn_drn_gis)
    # add more drainage structures
    model_config.drainage_structures.create(fn_drn_gis, merge=True)
    assert (
        len(model_config.drainage_structures.data.index) == nr_drainage_structures * 2
    )


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_storage_volume(tmp_dir, case):
    # create two helper functions to get regular/quadtree components
    def get_grid(mod):
        return mod.grid if case == "test1" else mod.quadtree_grid

    def get_storage_component(mod):
        return mod.storage_volume if case == "test1" else mod.quadtree_storage_volume

    # define the roots of the models
    root = join(TESTDATADIR, _cases[case]["example"])
    tmp_root = join(tmp_dir, "storage_volume_test")

    # create two aribitrary polygons and a point
    coords1 = [
        (318000.0, 5043000.0),
        (321000.0, 5043000.0),
        (321000.0, 5045500.0),
        (318000.0, 5045500.0),
        (318000.0, 5043000.0),
    ]
    poly1 = Polygon(coords1)
    coords2 = [
        (320500.0, 5044500.0),
        (321500.0, 5044500.0),
        (321500.0, 5046000.0),
        (320500.0, 5046000.0),
        (320500.0, 5044500.0),
    ]
    poly2 = Polygon(coords2)

    # create a geodataframe with the two polygons
    gdf = gpd.GeoDataFrame({"geometry": [poly1, poly2]}, crs=32633)
    gdf["volume"] = [None, 1000]

    # also create an arbitrary point
    point = Point(320000, 5044000)
    point_gdf = gpd.GeoDataFrame({"geometry": [point]}, crs=32633)
    point_gdf["volume"] = 20

    # read the sfincs model and change the root
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.root.set(tmp_root, mode="w+")

    # use correct storage component
    storage = get_storage_component(mod)
    grid = get_grid(mod)

    # test setup_storage_volume with polygons
    # one polygon has no volume specifed, the other has a volume of 1000
    # the non-specified gets the volume of the input argument
    storage.create(storage_locs=gdf, volume=10000)
    assert grid.data["vol"].sum() == 11000

    # test setup_storage_volume with points
    storage.create(storage_locs=point_gdf, merge=True)
    assert grid.data["vol"].sum() == 11020

    # write the model to test IO
    mod.write()

    # read the model again
    mod1 = SfincsModel(root=tmp_root, mode="r")
    mod1.config.read()

    # again get right component
    grid1 = get_grid(mod1)
    grid1.read(data_vars=["vol"])

    # now compare the storage volumes
    if case == "test1":
        assert np.isclose(
            mod1.grid.data["vol"].raster.mask_nodata().sum().values
            - mod.grid.data["vol"].sum().values,
            0,
        )
    elif case == "test2":
        assert np.isclose(
            (mod1.quadtree_grid.data["vol"] - mod.quadtree_grid.data["vol"]).sum(), 0
        )

    # now redo the tests with a rotated grid for the regular grid only
    if case == "test1":
        config = mod.config.data.model_copy()
        mod = SfincsModel(root=tmp_root, mode="w+")

        # get the config from the first model and add a rotation
        config.__setattr__("rotation", 10)
        mod.config._data = config
        mod.grid.update_grid_from_config()

        # test setup_storage_volume with
        # drop volume column from gdf
        gdf = gdf.drop(columns=["volume"])
        mod.storage_volume.create(storage_locs=gdf, volume=[350, 800])

        # check if the volumes are correct
        assert np.isclose(mod.grid.data["vol"].sum(), 1150)

        # drop volume column from gdf
        point_gdf = point_gdf.drop(columns=["volume"])
        mod.storage_volume.create(storage_locs=point_gdf, volume=34.5, merge=False)

        assert np.isclose(mod.grid.data["vol"].sum(), 34.5)

        # check index of the point with maximum volume
        index = mod.grid.data["vol"].argmax()
        assert index == 2113


def test_observations(model_config, tmp_dir):
    # read
    model_config.observation_points.read()
    model_config.cross_sections.read()

    # observation points
    assert not model_config.observation_points.data.empty
    nr_observation_points = len(model_config.observation_points.data.index)
    # write obs file only
    tmp_root = tmp_dir / "observation_points_test"
    model_config.root.set(tmp_root, mode="w+")
    model_config.observation_points.write()
    assert isfile(join(model_config.root.path, "sfincs.obs"))
    assert not isfile(join(model_config.root.path, "sfincs.crs"))
    fn_obs_gis = join(model_config.root.path, "gis", "obs.geojson")
    assert isfile(fn_obs_gis)
    # add more observation points
    model_config.observation_points.create(fn_obs_gis, merge=True)
    assert len(model_config.observation_points.data.index) == nr_observation_points * 2

    # observation lines
    assert not model_config.cross_sections.data.empty
    nr_observation_lines = len(model_config.cross_sections.data.index)
    # write crs file only
    tmp_root = tmp_dir / "observation_lines_test"
    model_config.root.set(tmp_root, mode="w+")
    model_config.cross_sections.write()
    assert isfile(join(model_config.root.path, "sfincs.crs"))
    assert not isfile(join(model_config.root.path, "sfincs.obs"))
    fn_crs_gis = join(model_config.root.path, "gis", "crs.geojson")
    assert isfile(fn_crs_gis)
    # add more observation lines
    model_config.cross_sections.create(fn_crs_gis, merge=True)
    assert len(model_config.cross_sections.data.index) == nr_observation_lines * 2


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_read_results(case):
    root = join(TESTDATADIR, _cases[case]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.output.read()
    assert all([v in mod.output.data for v in ["zs", "zsmax", "inp"]])


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_plots(case, tmpdir):
    root = join(TESTDATADIR, _cases[case]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.plot_forcing(fn_out=join(tmpdir, "forcing.png"))
    assert isfile(join(tmpdir, "forcing.png"))
    fn_out = join(tmpdir, "basemap.png")
    if case == "test2":
        mod.plot_basemap(
            fn_out=fn_out,
            bmap="sat",
            plot_bounds=False,  # does not work yet for quadtree
        )
    else:
        mod.plot_basemap(fn_out=fn_out, bmap="sat")
    assert isfile(fn_out)
