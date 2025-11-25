| Old API | New API | Argument Mapping |
|---------|---------|----------------|
| `SfincsModel.setup_grid()` | `SfincsModel.grid.create()` | unchanged |
| `SfincsModel.setup_grid_from_region()` | `SfincsModel.grid.create_from_region()` | unchanged |
| `SfincsModel.setup_dep()` | `SfincsModel.elevation.create()` | datasets_dep → elevation_list, elevtn → elevation |
| `SfincsModel.setup_mask_active()` | `SfincsModel.mask.create_active()` | mask → removed, mask_buffer → removed, include_mask → include_polygon, extra_option1 → include_zmin, extra_option2 → include_zmax, exclude_mask → exclude_polygon, extra_option3 → exclude_zmin, extra_option4 → exclude_zmax |
| `SfincsModel.setup_mask_bounds()` | `SfincsModel.mask.create_boundary()` | include_mask → include_polygon, extra_option1 → include_zmin, extra_option2 → include_zmax, exclude_mask → exclude_polygon, extra_option3 → exclude_zmin, extra_option4 → exclude_zmax |
| `SfincsModel.setup_subgrid()` | `SfincsModel.subgrid.create()` | datasets_dep → elevation_list, datasets_rgh → roughness_list, datasets_riv → river_list, nr_levels → nlevels |
| `SfincsModel.setup_river_inflow()` | `SfincsModel.river.create_inflow()` | unchanged |
| `SfincsModel.setup_river_outflow()` | `TODO()` | unchanged |
| `SfincsModel.setup_constant_infiltration()` | `SfincsModel.infiltration.create_constant()` | unchanged |
| `SfincsModel.setup_cn_infiltration()` | `SfincsModel.infiltration.create_cn()` | unchanged |
| `SfincsModel.setup_cn_infiltration_with_ks()` | `SfincsModel.infiltration.create_cn_with_recovery()` | unchanged |
| `SfincsModel.setup_manning_roughness()` | `SfincsModel.roughness.create()` | datasets_rgh → roughness_list |
| `SfincsModel.setup_observation_points()` | `SfincsModel.observation_points.create()` | unchanged |
| `SfincsModel.setup_observation_lines()` | `SfincsModel.cross_sections.create()` | unchanged |
| `SfincsModel.setup_structures()` | `SfincsModel.weirs.create()` | structures → locations, stype → removed |
| `SfincsModel.setup_structures()` | `SfincsModel.thin_dams.create()` | structures → locations, stype → removed |
| `SfincsModel.setup_drainage_structures()` | `SfincsModel.drainage_structures.create()` | structures → locations |
| `SfincsModel.setup_storage_volume()` | `SfincsModel.storage_volume.create()` | unchanged |
| `SfincsModel.setup_waterlevel_forcing()` | `SfincsModel.water_level.create()` | unchanged |
| `SfincsModel.setup_waterlevel_bnd_from_mask()` | `SfincsModel.water_level.create_boundary_points_from_mask()` | distance → bnd_dist, new_option → min_dist |
| `SficnsModel.setup_discharge_forcing()` | `SfincsModel.discharge_points.create()` | unchanged |
| `SfincsModel.setup_discharge_forcing_from_grid()` | `TODO()` | unchanged |
| `SfincsModel.setup_precip_forcing_from_grid()` | `SfincsModel.precipitation.create()` | unchanged |
| `SfincsModel.setup_precip_forcing()` | `SfincsModel.precipitation.create_uniform()` | unchanged |
| `SficsModel.setup_pressure_forcing_from_grid()` | `SfincsModel.pressure.create()` | unchanged |
| `SfincsModel.setup_wind_forcing_from_grid()` | `SfincsModel.wind.create()` | unchanged |
| `SfincsModel.setup_wind_forcing()` | `SfincsModel.wind.create_uniform()` | unchanged |
| `SfincsModel.setup_config()` | `SfincsModel.config.update()` | unchanged |
| `SfincsModel.plot_basemap()` | `SfincsModel.plot_basemap()` | unchanged |
| `SfincsModel.plot_forcing()` | `SfincsModel.plot_forcing()` | unchanged |
| `SfincsModel.read_results()` | `SfincsModel.output.read()` | unchanged |
