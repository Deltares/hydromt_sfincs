"""Test sfincs model class against hydromt.models.model_api"""

from os.path import isfile, join

import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, Point
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from hydromt.cli._utils import parse_config

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


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_model_class(case):
    # read model in examples folder
    root = join(TESTDATADIR, _cases[case]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    # run test_model_api() method
    non_compliant_list = mod._test_model_api()
    # drop non-compliant variables with "results" and "mesh" in name
    non_compliant_list = [
        v for v in non_compliant_list if "results" not in v and "mesh" not in v
    ]
    assert len(non_compliant_list) == 0


def test_states(mod):
    # create dummy state and set to states
    mask = mod.grid["dep"] < -0.5
    zsini = xr.where(mask, 0.5, -9999.0)
    zsini.raster.set_nodata(-9999.0)
    zsini.raster.set_crs(mod.crs)
    mod.set_states(zsini, "zsini")
    # write and check if isfile
    mod.write_grid()  # required to write file
    mod.write_states()
    mod.write_config()
    assert isfile(join(mod.root, "sfincs.zsini"))
    # read and check if identical
    mod1 = SfincsModel(root=mod.root, mode="r")
    assert np.allclose(mod1.states["zsini"], mod.states["zsini"])


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
    hsg = xr.where(model.grid.data["dep"] < 2, 1, 3)
    ksat = xr.where(model.grid.data["dep"] < 1, 0.01, 0.2)
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
        datasets_dep=[
            {"elevtn": "merit_hydro", "zmin": 0.001},
            {"elevtn": "gebco"},
        ],
        datasets_rgh=[
            {
                "lulc": "vito_2015",
                "reclass_table": join(TESTDATADIR, "local_data", "vito_mapping.csv"),
            }
        ],
        datasets_riv=[
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


def test_structs(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    # read
    mod.set_config("thdfile", "sfincs.thd")
    mod.read_grid()
    mod.read_geoms()
    assert "thd" in mod.geoms
    # write thd file only
    tmp_root = str(tmpdir.join("struct_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["thd"])
    assert isfile(join(mod.root, "sfincs.thd"))
    assert not isfile(join(mod.root, "sfincs.obs"))
    fn_thd_gis = join(mod.root, "gis", "thd.geojson")
    assert isfile(fn_thd_gis)
    # add second thd file
    mod.setup_structures(fn_thd_gis, stype="thd")
    assert len(mod.geoms["thd"].index) == 2
    # setup weir file from thd.geojson using dz option
    with pytest.raises(ValueError, match="Weir structure requires z"):
        mod.setup_structures(fn_thd_gis, stype="weir")
    mod.setup_structures(fn_thd_gis, stype="weir", dz=2)
    assert "weir" in mod.geoms
    assert "weirfile" in mod.config
    mod.write_geoms()
    assert isfile(join(mod.root, "sfincs.weir"))
    # test with buffer
    mod.setup_structures(fn_thd_gis, stype="weir", buffer=5, dep="dep", merge=False)
    assert len(mod.geoms["weir"].index) == 2


def test_drainage_structures(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    # read
    mod.set_config("drnfile", "sfincs.drn")
    mod.read_grid()
    mod.read_geoms()
    assert "drn" in mod.geoms
    nr_drainage_structures = len(mod.geoms["drn"].index)
    # write drn file only
    tmp_root = str(tmpdir.join("drainage_struct_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["drn"])
    assert isfile(join(mod.root, "sfincs.drn"))
    assert not isfile(join(mod.root, "sfincs.obs"))
    fn_drn_gis = join(mod.root, "gis", "drn.geojson")
    assert isfile(fn_drn_gis)
    # add more drainage structures
    mod.setup_drainage_structures(fn_drn_gis, merge=True)
    assert len(mod.geoms["drn"].index) == nr_drainage_structures * 2


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_storage_volume(tmp_dir, case):
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
    mod.set_root(tmp_root, mode="w+")

    # test setup_storage_volume with polygons
    # one polygon has no volume specifed, the other has a volume of 1000
    # the non-specified gets the volume of the input argument
    mod.setup_storage_volume(storage_locs=gdf, volume=10000)

    if case == "test1":
        assert mod.grid["vol"].sum() == 11000
    elif case == "test2":
        assert mod.quadtree.data["vol"].sum() == 11000

    # test setup_storage_volume with points
    mod.setup_storage_volume(storage_locs=point_gdf, merge=True)

    if case == "test1":
        assert mod.grid["vol"].sum() == 11020
    elif case == "test2":
        assert mod.quadtree.data["vol"].sum() == 11020

    # write the model to test IO
    mod.write()

    # read the model again
    mod1 = SfincsModel(root=tmp_root, mode="r")
    mod1.read_config()
    mod1.read_grid(data_vars=["vol"])

    # now compare the storage volumes
    if case == "test1":
        assert np.isclose(
            mod1.grid["vol"].raster.mask_nodata().sum().values
            - mod.grid["vol"].sum().values,
            0,
        )
    elif case == "test2":
        assert np.isclose(
            (mod1.quadtree.data["vol"] - mod.quadtree.data["vol"]).sum(), 0
        )

    # now redo the tests with a rotated grid for the regular grid only
    if case == "test1":
        config = mod.config.copy()
        mod = SfincsModel(root=tmp_root, mode="w+")

        # get the config from the first model and add a rotation
        config["rotation"] = 10
        mod.config.update(config)
        mod.update_grid_from_config()

        # test setup_storage_volume with
        # drop volume column from gdf
        gdf = gdf.drop(columns=["volume"])
        mod.setup_storage_volume(storage_locs=gdf, volume=[350, 800])

        # check if the volumes are correct
        assert np.isclose(mod.grid["vol"].sum(), 1150)

        # drop volume column from gdf
        point_gdf = point_gdf.drop(columns=["volume"])
        mod.setup_storage_volume(storage_locs=point_gdf, volume=34.5, merge=False)

        assert np.isclose(mod.grid["vol"].sum(), 34.5)

        # check index of the point with maximum volume
        index = mod.grid["vol"].argmax()
        assert index == 2113


def test_observations(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r+")
    # read
    mod.set_config("obsfile", "sfincs.obs")
    mod.read_grid()
    mod.read_geoms()

    # observation points
    assert "obs" in mod.geoms
    nr_observation_points = len(mod.geoms["obs"].index)
    # write obs file only
    tmp_root = str(tmpdir.join("observation_points_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["obs"])
    assert isfile(join(mod.root, "sfincs.obs"))
    assert not isfile(join(mod.root, "sfincs.crs"))
    fn_obs_gis = join(mod.root, "gis", "obs.geojson")
    assert isfile(fn_obs_gis)
    # add more observation points
    mod.setup_observation_points(fn_obs_gis, merge=True)
    assert len(mod.geoms["obs"].index) == nr_observation_points * 2

    # observation lines
    assert "crs" in mod.geoms
    nr_observation_lines = len(mod.geoms["crs"].index)
    # write crs file only
    tmp_root = str(tmpdir.join("observation_lines_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["crs"])
    assert isfile(join(mod.root, "sfincs.crs"))
    assert not isfile(join(mod.root, "sfincs.obs"))
    fn_crs_gis = join(mod.root, "gis", "crs.geojson")
    assert isfile(fn_crs_gis)
    # add more observation lines
    mod.setup_observation_lines(fn_crs_gis, merge=True)
    assert len(mod.geoms["crs"].index) == nr_observation_lines * 2


def test_forcing_io(tmp_dir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    # read
    mod.read_forcing()

    # write forcing
    tmp_root = join(tmp_dir, "forcing_test")
    mod.set_root(tmp_root, mode="w")
    mod.write_forcing()
    mod.write_config()

    # read and check if identical
    mod1 = SfincsModel(root=tmp_root, mode="r")
    mod1.read_forcing()

    # for all forcing variables, check if they are identical
    for key in mod.forcing.keys():
        assert np.allclose(mod1.forcing[key].values, mod.forcing[key].values)

    # now change the timeseries-format and write again
    tmp_root = join(tmp_dir, "forcing_test2")
    mod1.set_root(tmp_root, mode="w+")
    mod1.write_forcing(fmt="%7.1f")
    mod1.write_config()

    # read and check if identical (only for bzs here)
    mod2 = SfincsModel(root=tmp_root, mode="r")
    mod2.read_forcing(data_vars=["waterlevel"])
    assert np.isclose(
        np.sum(mod2.forcing["bzs"].values - mod1.forcing["bzs"].values), 0.73
    )


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_read_results(case):
    root = join(TESTDATADIR, _cases[case]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.read_results()
    assert all([v in mod.results for v in ["zs", "zsmax", "inp"]])


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


@pytest.mark.parametrize("case", list(_cases.keys())[:1])
def test_model_build(tmpdir, case):
    # compare results with model from examples folder
    root = str(tmpdir.join(case))
    root0 = TESTMODELDIR

    # Build model
    ini_fn = join(TESTDATADIR, _cases[case]["ini"])
    opt = parse_config(ini_fn)
    # logger = setuplog(path=join(root, "hydromt.log"), log_level=10)
    mod1 = SfincsModel(root=root, mode="w", **opt.pop("global", {}))
    mod1.build(opt=opt)
    # Check if model is api compliant
    non_compliant_list = mod1.test_model_api()
    assert len(non_compliant_list) == 0

    # read and compare with model from examples folder
    mod0 = SfincsModel(root=root0, mode="r")
    mod0.read()
    mod1 = SfincsModel(root=root, mode="r")
    mod1.read()
    # TODO using hydromt core Model._check_equal after fix https://github.com/Deltares/hydromt/issues/253
    # check config
    if mod0.config:
        assert mod0.config == mod1.config, "config mismatch"
    # check maps
    invalid_maps = []
    if len(mod0.grid) > 0:
        assert np.all(mod0.crs == mod1.crs), "map crs"
        mask = (mod0.grid["msk"] > 0).values  # compare only active cells
        mask1 = (mod1.grid["msk"] > 0).values
        assert np.allclose(mask, mask1), "mask mismatch"
        for name in mod0.grid.raster.vars:
            if name == "msk":
                continue
            map0 = mod0.grid[name].values
            map1 = mod1.grid[name].values
            if not np.allclose(map0[mask], map1[mask]):
                invalid_maps.append(name)
    invalid_map_str = ", ".join(invalid_maps)
    assert len(invalid_maps) == 0, f"invalid maps: {invalid_map_str}"
    # check geoms
    invalid_geoms = []
    if mod0.geoms:
        for name in mod0.geoms:
            try:
                assert_geodataframe_equal(
                    mod0.geoms[name],
                    mod1.geoms[name],
                    check_less_precise=True,  # allow for rounding errors in geoms
                    check_like=True,  # order may be different
                    check_geom_type=True,  # geometry types should be the same
                    normalize=True,  # normalize geometry
                )
            except AssertionError:  # re-raise error with geom name
                invalid_geoms.append(name)
    assert len(invalid_geoms) == 0, f"invalid geoms: {invalid_geoms}"
    # check forcing
    if mod0.forcing:
        for name in mod0.forcing:
            assert np.allclose(
                mod0.forcing[name], mod1.forcing[name]
            ), f" invalid forcing: {name}"
