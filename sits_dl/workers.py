from typing import List, Union, Optional
import multiprocessing as pmp
import rioxarray as rxr
from xarray import Dataset, DataArray
import numpy as np

from sits_dl.forestmask import ForestMask

def reader(tile_height: int, tile_width: int, n_bands: int, sequence: int, rstep: int, cstep:int, cube_inputs: List, forest_mask: ForestMask, q: pmp.Queue) -> None:
    for row in range(0, tile_height, rstep):
        for col in range(0, tile_width, cstep):
            mask: Optional[np.ndarray] = forest_mask.get_mask(row, row + rstep, col, col + cstep)

            s2_cube_np: np.ndarray = np.full(
                (sequence, n_bands, rstep, cstep),
                np.nan,
                dtype=np.float32
            )

            index_offset = 0 if (diff := sequence - len(cube_inputs)) < 0 else abs(diff)
            if index_offset == 0:
                cube_inputs = cube_inputs[-diff:]
            for _, (index, cube_input) in zip(range(sequence), enumerate(cube_inputs, start=index_offset)):
                """
                Zipping by sequences ensures that at mose sequence_length many observations are read.
                This makes padding further down below unnecessary!
                """
                ds: Union[Dataset, DataArray] = rxr.open_rasterio(cube_input)
                clipped_ds = ds.isel(y=slice(row, row + rstep), x=slice(col, col + cstep))
                s2_cube_np[index] = clipped_ds.to_numpy()
                ds.close()
                del ds, clipped_ds
            
            q.put((s2_cube_np, mask, index_offset))
            del s2_cube_np, mask, index_offset
