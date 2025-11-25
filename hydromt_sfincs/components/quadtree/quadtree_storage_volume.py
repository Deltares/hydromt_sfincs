import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import xugrid as xu

from hydromt.model.components import ModelComponent

from hydromt_sfincs import workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {"vol": {"standard_name": "storage volume", "unit": "m3"}}


class SfincsQuadtreeStorageVolume(ModelComponent):
    """SFINCS storage volume component for quadtree grids."""

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.quadtree_grid.data["mask"]
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.quadtree_grid.mask

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        # The mask values are written when the quadtree grid is written
        pass

    def create(
        self,
        storage_locs: Union[str, Path, gpd.GeoDataFrame],
        volume: Union[float, List[float]] = None,
        height: Union[float, List[float]] = None,
        merge: bool = True,
    ):
        """Setup storage volume.

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
            da_vol = xu.full_like(self.mask, 0, dtype=np.float64)

        # add storage volumes form gdf to da_vol
        da_vol = workflows.add_storage_volume_qt(
            da_vol,
            gdf,
            volume=volume,
            height=height,
            logger=logger,
        )

        # set grid
        mname = "vol"
        da_vol.attrs.update(**_ATTRS.get(mname, {}))
        self.model.quadtree_grid.set(da_vol, name=mname)
        # update config
        self.model.config.set(f"{mname}file", f"sfincs.{mname[:3]}")
