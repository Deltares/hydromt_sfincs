import logging

import numpy as np

logger = logging.getLogger(__name__)


class SfincsRegularGridMixin:
    """Mixin to process regular grids in blocks."""

    def compute_regular_grid(
        self,
        compute_block,
        outputs: dict,
        block_size: int = 2000,
        source_resolution: float | None = None,
    ):
        """Traverse regular-grid blocks, compute values, and stitch outputs."""
        x_dim, y_dim = self.mask.raster.x_dim, self.mask.raster.y_dim
        nmax = self.mask.sizes[y_dim]
        mmax = self.mask.sizes[x_dim]

        if source_resolution is None:
            block_cells = block_size
        else:
            grid_resolution = np.mean(
                [abs(self.mask.raster.res[0]), abs(self.mask.raster.res[1])]
            )
            if self.model.crs.is_geographic:
                grid_resolution *= 111111.0
            refi = grid_resolution / source_resolution
            block_cells = int(np.floor(block_size / refi))
        block_cells = max(block_cells, 2)

        n_blocks = int(np.ceil(nmax / block_cells))
        m_blocks = int(np.ceil(mmax / block_cells))

        if mmax % block_cells == 1:
            m_blocks -= 1
        if nmax % block_cells == 1:
            n_blocks -= 1

        logger.info(
            f"Processing regular grid in {n_blocks}x{m_blocks} blocks "
            f"using '{getattr(compute_block, '__name__', repr(compute_block))}'"
        )

        for ii in range(m_blocks):
            bm0 = ii * block_cells
            bm1 = min(bm0 + block_cells, mmax)
            if ii == m_blocks - 1:
                bm1 = mmax

            for jj in range(n_blocks):
                bn0 = jj * block_cells
                bn1 = min(bn0 + block_cells, nmax)
                if jj == n_blocks - 1:
                    bn1 = nmax

                logger.debug(
                    f"block {jj + ii * n_blocks + 1}/{n_blocks * m_blocks} -- "
                    f"col {bm0}:{bm1 - 1} | row {bn0}:{bn1 - 1}"
                )

                block = self.mask.isel(
                    {x_dim: slice(bm0, bm1), y_dim: slice(bn0, bn1)}
                ).load()
                block_outputs = compute_block(block)
                if not isinstance(block_outputs, dict):
                    block_outputs = dict(zip(outputs, block_outputs))

                selection = {x_dim: slice(bm0, bm1), y_dim: slice(bn0, bn1)}
                for name, da_block in block_outputs.items():
                    outputs[name][selection] = da_block

        return outputs
