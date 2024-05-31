from pathlib import Path
from typing import Optional
import rioxarray as rxr
import numpy as np
import xarray

class ForestMask():
    def __init__(self, parent: Optional[Path], tile: str, mask_glob: str) -> None:
        self.parent: Optional[Path] = parent
        self.tile: str = tile
        self.mask_glob: str = mask_glob

    def get_mask(self, rs: int, re: int, cs: int, ce: int) -> Optional[np.ndarray]:
        if self.parent is None:
            return None
        
        mask_dir: Path = self.parent / self.tile
        if not mask_dir.exists():
            mask_dir = mask_dir.parent
        try:
            mask_path: str = str(list(mask_dir.glob(self.mask_glob)).pop())
        except IndexError:
            raise RuntimeError("Provided mask directory and name but could not find any files")

        with rxr.open_rasterio(mask_path) as ds:
            mask_ds: xarray.Dataset = ds.isel(
                y=slice(rs, re), x=slice(cs, ce)
            )
            mask = np.array(mask_ds, ndmin=2, dtype=np.bool_).squeeze(axis=0).reshape((-1,))
            del mask_ds
        
        return mask
