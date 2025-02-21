# %% Simple example grid of building a SFINCS model wiith V1, using supported functions
from os.path import join
from datetime import datetime
import sys

sys.path.append(r"c:\Users\leijnse\repos\hydromt_sfincs")

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

# %% Initialize model
sfincs_root = r"./testmodel_V1"

sf = SfincsModel(root=sfincs_root, mode="w+")

# %% Create grid from region geojson
region_fn = join("data", "include_polygon_V1.geojson")

sf.grid.create_from_region(
    region={"geom": region_fn}, res=100, crs=25832, rotated=False
)

# %% plot
sf.region.plot()

# %% Add observation points
obs_fn = join("data", "obs.geojson")

gdf_obs = sf.data_catalog.get_geodataframe(obs_fn)

sf.observation_points.create(locations=gdf_obs, merge=False)

# %% plot
sf.observation_points.data.plot()

# %% Set some parameters
# update the configuration with new values
inpdict = {
    "tref": datetime(2023, 10, 18, 00, 00, 00),
    "tstart": datetime(2023, 10, 18, 00, 00, 00),
    "tstop": datetime(2023, 10, 22, 00, 00, 00),
}

# This does not work, because we said that that we expect a datetime:
# inpdict = {
#     "tref": "20231018 000000",
#     "tstart": "20231018 000000",
#     "tstop": "20231022 000000",
# }

sf.config.update(inpdict)

# %% Write model
# sf.grid.write() #> haven't make msk&dep yet, so does nothing currently
# %%
sf.observation_points.write(join(sf.root.path, "sfincs.obs"))
# sf.observation_points.write() #FIXME - think of correct behaviour with root and _filename

# %%
sf.config.write()

# %%
# sf.write() - TODO - later we want to have this working again
