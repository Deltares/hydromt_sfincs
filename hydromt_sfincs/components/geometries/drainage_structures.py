import geopandas as gpd

from hydromt.model.components import ModelComponent
from hydromt.model import Model


class SfincsDrainageStructures(ModelComponent):
    def __init__(
        self,
        model: Model,
    ):
        self._filename: str = "sfincs.drain"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Drainage structures data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data


# %% Original HydroMT-SFINCS setup_ functions:
#   setup_drainage_structures

# %% core HydroMT-SFINCS functions:
# _initialize
# read
# write
# set
# create
# add
# delete
# clear

# %% DDB GUI focused additional functions:
# - yet unsupported in DDB-
