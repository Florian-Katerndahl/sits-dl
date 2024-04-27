from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from re import search, Pattern, compile
import torch
from torch.utils.data import TensorDataset, DataLoader
import rasterio
import numpy as np
import rioxarray as rxr
from xarray import Dataset, DataArray


class Models(Enum):
    """
    Known model types.
    """
    UNDEFINED = 0
    LSTM = 1
    TRANSFORMER = 2
    SBERT = 3


class TensorDataCube:
    """
    Class for creating and iterating over a single FORCE tile converted to a pytorch tensor.
    """

    OUTPUT_NODATA: int = 255
    DATE_IN_FPATH: Pattern = compile(r"(?<=/)\d{8}(?=_)")

    def __init__(
        self,
        input_directory: Path,
        tile_id: str,
        input_glob: str,
        start: Optional[int],
        cutoff: int,
        inference_type: Models,
        row_step: Optional[int] = None,
        column_step: Optional[int] = None,
        sequence_length: Optional[int] = None,
    ) -> "TensorDataCube":
        self.input_directory: Path = input_directory
        self.tile_id: str = tile_id
        self.input_glob: str = input_glob
        self.start: Optional[int] = start
        self.cutoff: int = cutoff
        self.inference_type: Models = inference_type
        self.row_step: Optional[int] = row_step
        self.column_step: Optional[int] = column_step
        self.sequence_length: Optional[int] = sequence_length

        self.cube_inputs: Optional[List[str]] = None
        self.image_info: Optional[Dict] = None
        self.output_metadata: Optional[Dict] = None

    def self_prep(self) -> None:
        tile_paths: List[str] = [str(p) for p in (self.input_directory / self.tile_id).glob(self.input_glob)]
        if self.start:
            self.cube_inputs = [
                tile_path for tile_path in tile_paths \
                    if self.start <= int(search(r"\d{8}", tile_path).group(0)) <= self.cutoff
            ]
        else:
            self.cube_inputs = [
                tile_path for tile_path in tile_paths \
                    if int(search(r"\d{8}", tile_path).group(0)) <= self.cutoff
            ]
        self.cube_inputs.sort()

        with rasterio.open(self.cube_inputs[0]) as f:
            self.output_metadata = f.meta
            input_bands, self.output_metadata["count"] = self.output_metadata["count"], 1
            self.output_metadata["dtype"] = rasterio.float32 if self.inference_type == Models.SBERT else rasterio.uint8
            self.output_metadata["nodata"] = TensorDataCube.OUTPUT_NODATA
            row_block, col_block = f.block_shapes[0]

        self.image_info = {
            "input_bands": input_bands,
            "tile_width": self.output_metadata["width"],
            "tile_height": self.output_metadata["height"],
        }

        self.row_step = self.row_step or row_block
        self.column_step = self.column_step or col_block

        if (self.image_info["tile_width"] % self.column_step) != 0 or (self.image_info["tile_height"] % self.row_step) != 0:
            raise AssertionError("Rows and columns must be divisible by their respective step sizes without remainder.")

    def __iter__(self):
        for row in range(0, self.image_info["tile_height"], self.row_step):
            for col in range(0, self.image_info["tile_width"], self.column_step):
                s2_cube_np: np.ndarray = np.empty(
                    (len(self.cube_inputs), self.image_info["input_bands"], self.row_step, self.column_step),
                    dtype=np.float32,
                )
                for index, cube_input in enumerate(self.cube_inputs):
                    ds: Union[Dataset, DataArray] = rxr.open_rasterio(cube_input)
                    clipped_ds = ds.isel(y=slice(row, row + self.row_step), x=slice(col, col + self.column_step))
                    s2_cube_np[index] = clipped_ds.to_numpy()
                    ds.close()
                    del clipped_ds

                # TODO broadcast arrays to (-1, obs, bands) before standardizing. Remember to reorder to (2,3,0,1) beforehand.
                # This would remove a couple of reshaping/transposing steps and might speed up the code
                if self.inference_type == Models.TRANSFORMER:
                    _t: Tuple[List[Union[datetime, float]], List[bool]] = TensorDataCube.pad_doy_sequence(
                        self.sequence_length,
                        self.cube_inputs,
                        self.inference_type
                    )
                    sensing_doys, sbert_mask = _t
                    sensing_doys_np: np.ndarray = np.array(sensing_doys)
                    sensing_doys_np = sensing_doys_np.reshape((self.sequence_length, 1, 1, 1))
                    sensing_doys_np = np.repeat(sensing_doys_np, self.row_step, axis=2)  # match actual data cube
                    sensing_doys_np = np.repeat(sensing_doys_np, self.column_step, axis=3)  # match actual data cube
                elif self.inference_type == Models.SBERT:
                    s2_cube_np = np.transpose(s2_cube_np, (2, 3, 0, 1))
                    # Goal: move all valid observations to the "front" of the data cube (i.e. lower indices)
                    # This is a "batch" operation of what Chris does sequentially
                    s2_cube_np = np.where(s2_cube_np == -9999, np.nan, s2_cube_np)
                    s2_cube_np = np.reshape(s2_cube_np, (-1, *s2_cube_np.shape[2:]))
                    pixels, observations, bands = s2_cube_np.shape
                    observations_without_nodata: np.ndarray = ~np.isnan(s2_cube_np).any(axis=2)
                    s2_cube_np[~observations_without_nodata] = np.nan
                    index_array = np.arange(observations).reshape((1, observations, 1)).repeat(pixels, axis=0)
                    index_array[observations_without_nodata] -= observations * 2
                    # sorting_keys is now an array where for each pixel, the indices are sorted such that all non-nan observations are followed by all nan observations.
                    # Within the respective groups, ordering is kept by observation date
                    sorting_keys = index_array.argsort(axis=1)
                    s2_cube_np = np.take_along_axis(s2_cube_np, sorting_keys, axis=1)  # actually move data

                    sbert_mask = ~np.isnan(s2_cube_np).any(axis=2).reshape((pixels, observations, 1))
                    sensing_dates_as_ordinal = [TensorDataCube.ordinal_observation(i) for i in self.cube_inputs]
                    sensing_doys_np = np.array(sensing_dates_as_ordinal).reshape((1, observations, 1)).repeat(pixels, axis=0)
                    sensing_doys_np = np.take_along_axis(sensing_doys_np, sorting_keys, axis=1)
                    doy_of_earliest_observations = TensorDataCube.toyday(sensing_doys_np[:, 0, 0]).reshape((pixels, 1)).repeat(observations, axis=1)
                    sensing_doys_np = (sensing_doys_np[..., 0] - sensing_doys_np[:, 0].repeat(observations, axis=1) + doy_of_earliest_observations).reshape((pixels, observations, 1))
                    sensing_doys_np[~sbert_mask] = 0
                    
                    # First reshape from -1, obs, bands to x, y, obs, bands to retain row order. Secondly, transpose to original order obs, bands, x, y for congruence with
                    # later code -- TO BE CHANGED
                    #
                    # at this point in time it's still possible for doys and true_observations to contain more or less observations than sequence length permits
                    s2_cube_np = np.reshape(s2_cube_np, (self.row_step, self.column_step, len(self.cube_inputs), self.image_info["input_bands"])).transpose((2,3,0,1))
                    # TODO check if sbert_mask and sensing_doy are correct!
                    sbert_mask = sbert_mask.astype(int).reshape((self.row_stel, self.column_step, len(self.cube_inputs), 1)).transpose((2,3,0,1))
                    sensing_doys_np = sensing_doys_np.reshape((self.row_step, self.column_step, len(self.cube_inputs), 1)).transpose((2,3,0,1))

                    sbert_mask = TensorDataCube.pad_datacube(self.sequence_length, sbert_mask, 0)
                    sensing_doys_np = TensorDataCube.pad_datacube(self.sequence_length, sensing_doys_np, 0)

                if self.inference_type in [Models.TRANSFORMER, Models.SBERT]:
                    # assumes s2_cube_np is padded with np.nan
                    s2_cube_np = TensorDataCube.pad_datacube(self.sequence_length, s2_cube_np)
                    s2_cube_np = (s2_cube_np - np.nanmean(s2_cube_np, axis=0)) / np.nanstd(s2_cube_np, axis=0) + 1e-6  # normalize across bands, dont touch DOYs
                    s2_cube_np = np.where(np.isnan(s2_cube_np), 0.0, s2_cube_np)
                    s2_cube_np = np.concatenate((s2_cube_np, sensing_doys_np), axis=1)

                s2_cube_npt: np.ndarray = np.transpose(s2_cube_np, (2, 3, 0, 1))
                s2_cube_torch: Union[torch.Tensor, torch.masked.masked_tensor] = torch.from_numpy(s2_cube_npt).float()

                if self.inference_type == Models.SBERT:
                    # such, that ugly transposing is not needed in main inference script
                    sbert_mask = np.transpose(sbert_mask, (2, 3, 0, 1))
                    s2_cube_torch = torch.cat([s2_cube_torch, torch.from_numpy(sbert_mask)], dim=3)

                try:
                    del s2_cube_np
                    del s2_cube_npt
                    del sensing_doys
                except UnboundLocalError:
                    pass

                yield s2_cube_torch, sbert_mask, row, col

    def __str__(self) -> str:
        pass

    def empty_output(self, dtype: torch.dtype=torch.long) -> torch.Tensor:
        return torch.full(
            [self.image_info["tile_height"], self.image_info["tile_width"]],
            fill_value=TensorDataCube.OUTPUT_NODATA,
            dtype=dtype,
        )
    

    @classmethod
    def to_dataloader(cls, chunk: torch.Tensor, batch_size: int, workers: int = 4) -> DataLoader:
        ds: TensorDataset = TensorDataset(torch.reshape(chunk, (-1, chunk.shape[2], chunk.shape[3])))  # TensorDataset splits along first dimension of input
        return DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=workers, persistent_workers=True)


    @classmethod
    def pad_doy_sequence(cls, target: int, observations: List[str], mtype: Models) -> Tuple[List[Union[datetime, float]], List[bool]]:
        diff: int = target - len(observations)
        true_observations: List[bool] = [True] * len(observations)
        if diff < 0:
            observations = observations[abs(diff):]  # deletes oldest entries first
            true_observations = true_observations[abs(diff):]
        elif diff > 0:
            observations += [0.0] * diff
            true_observations += [False] * diff

        # TODO remove assertion for "production"
        assert target == len(observations)

        return [TensorDataCube.fp_to_doy(i) for i in observations if isinstance(i, str)], true_observations


    @classmethod
    def pad_datacube(cls, target: int, datacube: np.ndarray, pad_value=np.nan) -> np.ndarray:
        diff: int = target - datacube.shape[0]
        if diff < 0:
            datacube = np.delete(datacube, list(range(abs(diff))), axis=0)  # deletes oldest entries first
        elif diff > 0:
            datacube = np.pad(datacube, ((0, diff), (0, 0), (0, 0), (0, 0)), constant_values=pad_value)

        # TODO remove assertion for "production"
        assert target == datacube.shape[0]

        return datacube

    @classmethod
    def fp_to_doy(cls, file_path: str, origin: Optional[str] = None) -> datetime:
        date_in_fp: Pattern = compile(r"(?<=/)\d{8}(?=_)")
        sensing_date: str = date_in_fp.findall(file_path)[0]
        d: datetime = datetime.strptime(sensing_date, "%Y%m%d")
        # This is a bit resource wasting since origin does not change but is re-computed for each
        # observation
        if origin:
            origin_date: str = date_in_fp.findall(origin)[0]
            od: datetime = datetime.strptime(origin_date, "%Y%m%d")
            return (d - od).days + od.timetuple().tm_yday
        return d.timetuple.tm_yday


    @staticmethod
    @np.vectorize(otypes=[int])
    def toyday(x):
        return datetime.fromordinal(x).timetuple().tm_yday
    

    @staticmethod
    def ordinal_observation(x: str) -> int:
        return datetime.strptime(TensorDataCube.DATE_IN_FPATH.findall(x)[0], "%Y%m%d").toordinal()
