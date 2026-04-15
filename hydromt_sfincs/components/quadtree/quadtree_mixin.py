import numpy as np
import logging

from hydromt_sfincs.utils import make_regular_grid

logger = logging.getLogger(__name__)


class SfincsQuadtreeMixin:
    """Mixin to process quadtree grids in chunks."""

    def compute_quadtree(self, compute_chunk, output, nrmax=2000, clip=None):
        """Traverse levels & chunks, automatically create DA, compute, and sample back.

        Parameters
        ----------
        compute_chunk : callable
            Function that computes values for a chunk. Should accept (da_like, ilev)
            and return a DataArray or array with computed values.
        output : array-like
            Output array to store results.
        nrmax : int, optional
            Maximum number of cells per chunk, by default 2000.
        clip : tuple, optional
            (min, max) values to clip output, by default None.
        """

        nlev = self.data.attrs["nr_levels"]
        xy = self.data.grid.face_coordinates
        dx, dy = self.data.attrs["dx"], self.data.attrs["dy"]

        level = self.data["level"].values - 1
        n = self.data["n"].values - 1
        m = self.data["m"].values - 1
        level_indices = [np.where(level == ilev)[0] for ilev in range(nlev)]

        func_name = getattr(compute_chunk, "__name__", repr(compute_chunk))
        logger.info(f"Computing quadtree values for {nlev} levels using '{func_name}'")

        # Ensure output is a dict for multiple variables
        if isinstance(output, (tuple, list)):
            output_dict = {i: arr for i, arr in enumerate(output)}
        elif isinstance(output, np.ndarray):
            output_dict = {0: output}
        elif isinstance(output, dict):
            output_dict = output
        else:
            raise ValueError("output must be dict or tuple/list of arrays")

        for ilev in range(nlev):
            idx = level_indices[ilev]
            if len(idx) == 0:
                continue

            xz, yz = xy[idx, 0], xy[idx, 1]
            n_level, m_level = n[idx], m[idx]
            dxmin, dymin = dx / 2**ilev, dy / 2**ilev

            x_min, x_max = xz.min() - dxmin, xz.max() + dxmin
            y_min, y_max = yz.min() - dymin, yz.max() + dymin
            x_chunks = np.arange(x_min, x_max, nrmax * dxmin)
            y_chunks = np.arange(y_min, y_max, nrmax * dymin)

            # Prepare storage for this level per output variable
            values_level_dict = {key: np.full(len(idx), np.nan) for key in output_dict}

            logger.info(
                f"Processing level {ilev+1} of {nlev} with {len(x_chunks)-1}x{len(y_chunks)-1} chunks"
            )

            for ix, x0 in enumerate(x_chunks):
                for iy, y0 in enumerate(y_chunks):
                    x1 = x_chunks[ix + 1] if ix < len(x_chunks) - 1 else x_max
                    y1 = y_chunks[iy + 1] if iy < len(y_chunks) - 1 else y_max

                    in_chunk = np.where(
                        (xz >= x0) & (xz < x1) & (yz >= y0) & (yz < y1)
                    )[0]
                    if len(in_chunk) == 0:
                        continue

                    # Make a regular grid for this chunk
                    da_like = make_regular_grid(
                        x0=self.data.attrs["x0"],
                        y0=self.data.attrs["y0"],
                        dx=dxmin,
                        dy=dymin,
                        mmax=m_level[in_chunk].max() + 1,
                        nmax=n_level[in_chunk].max() + 1,
                        rotation=self.data.attrs["rotation"],
                        crs=self.model.crs,
                        mmin=m_level[in_chunk].min(),
                        nmin=n_level[in_chunk].min(),
                        make_ugrid=False,
                    )

                    # Compute chunk values
                    da_outs = compute_chunk(da_like, ilev=ilev)

                    # If single output, convert to tuple
                    if not isinstance(da_outs, (tuple, list)):
                        da_outs = (da_outs,)

                    # Assign each output variable
                    for i, key in enumerate(values_level_dict):
                        da_out = da_outs[i]
                        idx_y = np.searchsorted(da_out.n.values, n_level[in_chunk])
                        idx_x = np.searchsorted(da_out.m.values, m_level[in_chunk])
                        values_level_dict[key][in_chunk] = da_out.values[idx_y, idx_x]

            # Clip if needed
            for key, vals in values_level_dict.items():
                if clip is not None:
                    if isinstance(clip, dict):
                        cmin, cmax = clip.get(key, (None, None))
                    else:
                        cmin, cmax = clip
                    if cmin is not None or cmax is not None:
                        vals = np.clip(vals, cmin, cmax)
                output_dict[key][idx] = vals
