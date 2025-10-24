# Mapping of old function names and argument changes to new function names and arguments
# For each function, we follow this structure:
# "old_function_name": {
#     "new_function": "new_function_name",
#     "args": {
#         "old_arg1": "new_arg1",
#         "old_arg2": None,  # argument removed
#         "old_arg3": "new_arg3"  # argument renamed
#     }
# }

conversion_map = {
    "SfincsModel.setup_grid": {
        "new_function": "SfincsModel.grid.create",
        "args": "same",  # all arguments remain the same
    },
    "SfincsModel.setup_grid_from_region": {
        "new_function": "SfincsModel.grid.create_from_region",
        "args": "same",  # all arguments remain the same
    },
    "SfincsModel.setup_dep": {
        "new_function": "SfincsModel.elevation.create",
        "args": {
            "datasets_dep": "elevation_sets",  # renamed argument
            # other arguments remain the same
        },
    },
    "SfincsModel.setup_mask_active": {
        "new_function": "SfincsModel.mask.create_active",
        "args": {
            "mask": None,  # argument removed, use include_polygon instead
            "mask_buffer": None,  # argument removed
            "include_mask": "include_polygon",
            "extra_option1": "include_zmin",
            "extra_option2": "include_zmax",
            "exclude_mask": "exclude_polygon",
            "extra_option3": "exclude_zmin",
            "extra_option4": "exclude_zmax",
        },
    },
    "SfincsModel.setup_mask_bounds": {
        "new_function": "SfincsModel.mask.create_boundary",
        "args": {
            "include_mask": "include_polygon",
            "extra_option1": "include_zmin",
            "extra_option2": "include_zmax",
            "exclude_mask": "exclude_polygon",
            "extra_option3": "exclude_zmin",
            "extra_option4": "exclude_zmax",
        },
    },
    "SfincsModel.setup_subgrid": {
        "new_function": "SfincsModel.subgrid.create",
        "args": {
            "datasets_dep": "elevation_sets",  # renamed argument
            "datasets_rgh": "roughness_sets",  # renamed argument
            "datasets_riv": "river_sets",  # renamed argument
            "nr_levels": "nlevels",  # renamed argument
        },
    },
    "SfincsModel.setup_river_inflow": {
        "new_function": "SfincsModel.river.create_inflow",
        "args": "same",  # all arguments remain the same
    },
    "SfincsModel.setup_river_outflow": {
        "new_function": None,  # TODO: not yet implemented,
        "args": "same",  # all arguments remain the same
    },
    "SfincsModel.setup_constant_infiltration": {
        "new_function": "SfincsModel.infiltration.create_constant",
        "args": "same",
    },
    "SfincsModel.setup_cn_infiltration": {
        "new_function": "SfincsModel.infiltration.create_cn",
        "args": "same",
    },
    "SfincsModel.setup_cn_infiltration_with_ks": {
        "new_function": "SfincsModel.infiltration.create_cn_with_recovery",
        "args": "same",
    },
    "SfincsModel.setup_manning_roughness": {
        "new_function": "SfincsModel.roughness.create",
        "args": {
            "datasets_rgh": "roughness_sets",  # renamed argument
            # other arguments remain the same
        },
    },
    "SfincsModel.setup_observation_points": {
        "new_function": "SfincsModel.observation_points.create",
        "args": "same",
    },
    "SfincsModel.setup_observation_lines": {
        "new_function": "SfincsModel.cross_sections.create",
        "args": "same",
    },
    "SfincsModel.setup_structures": {
        "new_functions": [  # list only used for splits
            {
                "name": "SfincsModel.weirs.create",
                "args": {"structures": "locations", "stype": None},
            },
            {
                "name": "SfincsModel.thin_dams.create",
                "args": {"structures": "locations", "stype": None},
            },
        ]
    },
    "SfincsModel.setup_drainage_structures": {
        "new_function": "SfincsModel.drainage_structures.create",
        "args": {
            "structures": "locations",
        },
    },
    "SfincsModel.setup_storage_volume": {
        "new_function": "SfincsModel.storage_volume.create",
        "args": "same",
    },
    "SfincsModel.setup_waterlevel_forcing": {
        "new_function": "SfincsModel.water_level.create",
        "args": "same",
    },
    "SfincsModel.setup_waterlevel_bnd_from_mask": {
        "new_function": "SfincsModel.water_level.create_boundary_points_from_mask",
        "args": {
            "distance": "bnd_dist",
            "new_option": "min_dist",
        },
    },
    "SficnsModel.setup_discharge_forcing": {
        "new_function": "SfincsModel.discharge_points.create",
        "args": "same",
    },
    "SfincsModel.setup_discharge_forcing_from_grid": {
        "new_function": None,  # TODO "SfincsModel.discharge_points.create_from_grid",
        "args": "same",
    },
    "SfincsModel.setup_precip_forcing_from_grid": {
        "new_function": "SfincsModel.precipitation.create",
        "args": "same",
    },
    "SfincsModel.setup_precip_forcing": {
        "new_function": "SfincsModel.precipitation.create_uniform",
        "args": "same",
    },
    "SficsModel.setup_pressure_forcing_from_grid": {
        "new_function": "SfincsModel.pressure.create",
        "args": "same",
    },
    "SfincsModel.setup_wind_forcing_from_grid": {
        "new_function": "SfincsModel.wind.create",
        "args": "same",
    },
    "SfincsModel.setup_wind_forcing": {
        "new_function": "SfincsModel.wind.create_uniform",
        "args": "same",
    },
    "SfincsModel.plot_basemap": {
        "new_function": "SfincsModel.plot_basemap",
        "args": "same",
    },
    "SfincsModel.plot_forcing": {
        "new_function": "SfincsModel.plot_forcing",
        "args": "same",
    },
}


def export_markdown(conversion_map, filename="conversion_table.md"):
    with open(filename, "w", encoding="utf-8") as f:  # <- add encoding
        f.write("| Old API | New API | Argument Mapping |\n")
        f.write("|---------|---------|----------------|\n")

        for old_func, info in conversion_map.items():
            entries = []

            if "new_function" in info:
                new_func = info["new_function"] or "TODO"
                args = info.get("args", "same")
                args_str = format_args(args)
                entries.append((old_func, new_func, args_str))

            elif "new_functions" in info:
                for nf in info["new_functions"]:
                    new_func = nf.get("name") or "TODO"
                    args = nf.get("args", "same")
                    args_str = format_args(args)
                    entries.append((old_func, new_func, args_str))

            for old, new, args_str in entries:
                f.write(f"| `{old}()` | `{new}()` | {args_str} |\n")


def format_args(args):
    if args == "same":
        return "unchanged"
    elif isinstance(args, dict):
        return ", ".join(
            f"{old_arg} → {('removed' if new_arg is None else new_arg)}"
            for old_arg, new_arg in args.items()
        )
    else:
        return str(args)


# Call the function, markdown table can be previewed by pressing Ctrl+Shift+V in VSCode
export_markdown(conversion_map, "conversion_table.md")
