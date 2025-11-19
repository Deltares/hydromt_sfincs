import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {"vol": {"standard_name": "storage volume", "unit": "m3"}}


class SfincsStorageVolume(ModelComponent):
    """SFINCS Storage Volume Component.

    This component contains methods to add storage volume data to the SFINCS model
    on regular grids. Storage volume can be used to represent the effect of green-
    infrastructure in urban environments, such as retention basins or rain barrels.

    .. note::
        The storage volume data is stored in the model grid's data dataset under the key "vol".

    See Also
    --------
    :py:class:`~hydromt_sfincs.components.grid.regulargrid.SfincsGrid`

    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["mask"]
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.grid.mask

    def read(self):
        """Not implemented, storage volume data is read when the grid is read."""
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        """Not implemented, storage volume data is written when the grid is written."""
        # The mask values are written when the quadtree grid is written
        pass

    @hydromt_step
    def create(
        self,
        storage_locs: Union[str, Path, gpd.GeoDataFrame],
        volume: Union[float, List[float]] = None,
        height: Union[float, List[float]] = None,
        merge: bool = True,
    ):
        """Create storage volume.

        Adds model layer:
        * **vol** map: storage volume for green infrastructure

        Parameters
        ----------
        storage_locs : str, Path
            Path, data source name, or geopandas object to storage location polygon or point geometry file.
            Optional "volume" or "height" attributes can be provided to set the storage volume.
        volume : float, optional
            Storage volume [m3], by default None
        height : float, optional
            Storage height [m], by default None
        merge : bool, optional
            If True, merge with existing storage volumes, by default True.

        """

        # read, clip and reproject
        gdf = self.data_catalog.get_geodataframe(
            storage_locs,
            geom=self.model.region,
            buffer=10,
        ).to_crs(self.model.crs)

        # if merge, add new storage volumes to existing ones
        if merge and "vol" in self.data:
            da_vol = self.data["vol"]
        else:
            da_vol = xr.full_like(self.mask, 0, dtype=np.float64)

        # add storage volumes form gdf to da_vol
        da_vol = workflows.add_storage_volume(
            da_vol,
            gdf,
            volume=volume,
            height=height,
            logger=logger,
        )

        # set grid
        mname = "vol"
        da_vol.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_vol, name=mname)
        # update config
        self.model.config.set(f"{mname}file", f"sfincs.{mname[:3]}")
