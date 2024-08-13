from datetime import datetime, date
from time import time
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from re import search, Pattern, compile
from warnings import warn
import torch
from torch.utils.data import TensorDataset, DataLoader
import rasterio
import numpy as np
from numba import njit, prange
import rioxarray as rxr
from xarray import Dataset, DataArray
from sits_dl.forestmask import ForestMask

@njit(['float32[:,:,:](float32[:,:,:], Omitted(1e-6))', 'float32[:,:,:](float32[:,:,:], float32)'], parallel=True)
def znorm_PSB(dc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply z-normalization to data cube per pixel and per band.

    .. note:: Only valid for a datacube with the shape (pixels, sequence, bands)

    :param dc: Input data cube
    :type dc: np.ndarray
    :param eps: Epsilon to add for numeric stability, defaults to 1e-6
    :type eps: float, optional
    :return: Datacube normalized for each pixel and band across sequence length.
    :rtype: np.ndarray
    """    
    out: np.ndarray = np.empty_like(dc)
    for pixel in prange(dc.shape[0]):
        for band in prange(dc.shape[-1]):
            out[pixel, :, band] = (dc[pixel, :, band] - np.nanmean(dc[pixel, :, band])) / np.nanstd(dc[pixel, :, band]) + eps
    return out


@torch.compile(dynamic=False, fullgraph=True)
def nanstd_psb_t(dc: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    return (dc - torch.nanmean(dc, dim=dim, keepdim=keepdim)).square().nanmean(dim=dim, keepdim=keepdim).sqrt()


@torch.compile(dynamic=False, fullgraph=True)
def znorm_psb_t(dc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _mean: torch.Tensor = torch.nanmean(dc, dim=1, keepdim=True)
    _std = (dc - _mean).square().nanmean(dim=1, keepdim=True).sqrt()
    normed = (dc - _mean) / _std + eps
    return normed


@torch.compile()
def preprocess_sbert(dc: np.ndarray, device: torch.device) -> torch.Tensor:
    dc_t: torch.Tensor = torch.from_numpy(dc).to(device=device).permute(2, 3, 0, 1).reshape((-1, *dc.shape[:2]))
    dc_t[dc_t == -9999] = torch.nan
    pixels, sequence_length, bands = dc_t.shape
    observations_without_nodata: torch.Tensor = ~dc_t.isnan().any(dim=2).to(device=device)
    dc_t[observations_without_nodata] = torch.nan
    index_array: torch.Tensor = torch.arange(sequence_length).reshape((1, sequence_length, 1)).repeat_interleave(pixels, dim=0).to(device=device)
    index_array[observations_without_nodata] -= sequence_length * 2
    sorting_keys: torch.Tensor = index_array.argsort(dim=1)
    dc_t = dc_t.take_along_dim(sorting_keys, dim=1)

    del pixels, sequence_length, bands, index_array
    # print(torch.mean(dc_t, dim=1), torch.std(dc_t, dim=1))

    return dc_t, sorting_keys


class Models(Enum):
    """
    Known model types.
    """
    LSTM = 0
    TRANSFORMER = 1
    SBERT = 2
    UNDEFINED = 99


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
        mask_dir: Optional[Path],
        mask_glob: str,
        device: torch.device,
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
        self.forestmask = ForestMask(mask_dir, tile_id, mask_glob)
        self.device = device

        self.cube_inputs: Optional[List[str]] = None
        self.image_info: Optional[Dict] = None
        self.output_metadata: Optional[Dict] = None

    def self_prep(self) -> None:
        warn(("Only front padding implemented currently. I.e. If DC is your datacube and n < seq, the first n-seq "
              "items are filled with NAs and only the last seq-many items are valid observations"))
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
                t_chunk = time()
                mask: Optional[np.ndarray] = self.forestmask.get_mask(row, row + self.row_step, col, col + self.column_step)

                s2_cube_np: np.ndarray = np.full(
                    (self.sequence_length, self.image_info["input_bands"], self.row_step, self.column_step),
                    np.nan,
                    dtype=np.float32
                )

                index_offset = 0 if (diff := self.sequence_length - len(self.cube_inputs)) < 0 else abs(diff)
                if index_offset == 0:
                    self.cube_inputs = self.cube_inputs[-diff:]
                for _, (index, cube_input) in zip(range(self.sequence_length), enumerate(self.cube_inputs, start=index_offset)):
                    """
                    Zipping by sequences ensures that at mose sequence_length many observations are read.
                    This makes padding further down below unnecessary!
                    """
                    ds: Union[Dataset, DataArray] = rxr.open_rasterio(cube_input)
                    clipped_ds = ds.isel(y=slice(row, row + self.row_step), x=slice(col, col + self.column_step))
                    s2_cube_np[index] = clipped_ds.to_numpy()
                    ds.close()
                    del ds, clipped_ds

                if self.inference_type == Models.TRANSFORMER:
                    _t: Tuple[List[Union[datetime, float]], List[bool]] = TensorDataCube.pad_doy_sequence(
                        self.sequence_length,
                        self.cube_inputs,
                        self.inference_type
                    )
                    sensing_doys, sbert_mask = _t
                    sensing_doys_np: np.ndarray = np.array(sensing_doys)
                    sensing_doys_np = sensing_doys_np.reshape((self.sequence_length, 1, 1, 1))
                    sensing_doys_np = np.repeat(sensing_doys_np, self.row_step, axis=2)     # match actual data cube
                    sensing_doys_np = np.repeat(sensing_doys_np, self.column_step, axis=3)  # match actual data cube

                    # lines below needed now for compatability with SBERT datacube
                    s2_cube_np = s2_cube_np.transpose((2, 3, 0, 1))
                    s2_cube_np = s2_cube_np.reshape((-1, *s2_cube_np.shape[2:]))
                    sensing_doys_np = sensing_doys_np.transpose((2, 3, 0, 1))
                    sensing_doys_np = sensing_doys_np.reshape((-1, *sensing_doys_np.shape[2:]))
                elif self.inference_type == Models.SBERT:
                    # Goal: move all valid observations to the "front" of the data cube (i.e. lower indices)
                    # TODO rewrite to use torch tensors??
                    # test_np = s2_cube_np.copy()
                    # t444 = time()
                    # s2_cube_np = np.transpose(s2_cube_np, (2, 3, 0, 1))
                    # s2_cube_np[s2_cube_np == -9999] = np.nan
                    # s2_cube_np = np.reshape(s2_cube_np, (-1, *s2_cube_np.shape[2:]))
                    # pixels, sequence_length, bands = s2_cube_np.shape
                    # observations_without_nodata: np.ndarray = ~np.isnan(s2_cube_np).any(axis=2)
                    # s2_cube_np[~observations_without_nodata] = np.nan
                    # index_array = np.arange(sequence_length).reshape((1, sequence_length, 1)).repeat(pixels, axis=0)
                    # index_array[observations_without_nodata] -= sequence_length * 2
                    # # sorting_keys is now an array where for each pixel, the indices are sorted such that all non-nan observations are followed by all nan observations.
                    # # Within the respective groups, ordering is kept by observation date
                    # sorting_keys = index_array.argsort(axis=1)
                    # s2_cube_np = np.take_along_axis(s2_cube_np, sorting_keys, axis=1)  # actually move data
                    # print(time() - t444)

                    s2_cube_np, sorting_keys = preprocess_sbert(s2_cube_np, self.device)
                    pixels, sequence_length, _ = s2_cube_np.shape
                    print(s2_cube_np.shape)

                    sbert_mask = ~s2_cube_np.isnan().any(dim=2).reshape((pixels, sequence_length, 1))

                    sensing_dates_as_ordinal: np.ndarray = np.zeros((sequence_length,), dtype=np.int32)
                    sensing_dates_as_ordinal[index_offset:] = [TensorDataCube.ordinal_observation(i) for i in self.cube_inputs]
                    sensing_doys_np = sensing_dates_as_ordinal.reshape((1, sequence_length, 1)).repeat(pixels, axis=0)
                    sensing_doys_np = np.take_along_axis(sensing_doys_np, sorting_keys.numpy(force=True), axis=1)
                    doy_of_earliest_observations = TensorDataCube.toyday(sensing_doys_np[:, 0, 0]).reshape((pixels, 1)).repeat(sequence_length, axis=1)
                    sensing_doys_np = (sensing_doys_np[..., 0] - sensing_doys_np[:, 0].repeat(sequence_length, axis=1) + doy_of_earliest_observations).reshape((pixels, sequence_length, 1))
                    sensing_doys_np[~sbert_mask.numpy(force=True)] = 0

                if self.inference_type in [Models.TRANSFORMER, Models.SBERT]:
                    s2_cube_np = znorm_psb_t(s2_cube_np)
                    s2_cube_np[s2_cube_np.isnan()] = 0.0

                if self.inference_type == Models.SBERT:
                    cube_and_doys = torch.cat([s2_cube_np, torch.from_numpy(sensing_doys_np).to(device=self.device), sbert_mask], dim=2)  # 11 seconds
                    del sbert_mask, sensing_dates_as_ordinal, sensing_doys_np, \
                        doy_of_earliest_observations, sorting_keys  # index_array, observations_without_nodata
                elif self.inference_type == Models.TRANSFORMER:
                    cube_and_doys = torch.cat([s2_cube_np, torch.from_numpy(sensing_doys_np).to(device=self.device)], dim=2)  # 6.7 seonds
                    del sensing_doys_no, sbert_mask, sensing_doys
                elif self.inference_type == Models.LSTM:
                    cube_and_doys = s2_cube_np
                
                # s2_cube_torch: torch.Tensor = torch.from_numpy(cube_and_doys).float()

                # del s2_cube_np, cube_and_doys

                yield cube_and_doys, mask, row, col, t_chunk


    def empty_output(self, dtype: torch.dtype=torch.long) -> torch.Tensor:
        return torch.full(
            [self.image_info["tile_height"], self.image_info["tile_width"]],
            fill_value=TensorDataCube.OUTPUT_NODATA,
            dtype=dtype,
            device="cpu"
        )
    

    @classmethod
    def to_dataloader(cls, chunk: torch.Tensor, batch_size: int, workers: int = 4) -> DataLoader:
        return DataLoader(chunk, batch_size=batch_size, pin_memory=True, num_workers=workers, persistent_workers=False)


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
            # TODO should also be abs(diff) - 1? is the offset by one even correct?
            #  Construction of DC above indicates that it's not correct!
            datacube = np.delete(datacube, list(range(abs(diff))), axis=0)  # deletes oldest entries first
        elif diff > 0:
            datacube = np.pad(datacube, ((0, diff), (0, 0), (0, 0), (0, 0)), constant_values=pad_value)  # should be end padding

        # TODO remove assertion for "production"
        assert target == datacube.shape[0]

        return datacube


    @classmethod
    def pad_long_datacube(cls, target: int, datacube: np.ndarray, pad_value: Union[int, float] = np.nan) -> np.ndarray:
        """Pad datacube in (x * y, observations, bands)-format in axis of observations.

        .. note:: Oldest items are deleted first, if datacube has more entries than sequence length.

        :param target: Sequence length
        :type target: int
        :param datacube: Datacube to pad or crop
        :type datacube: np.ndarray
        :param pad_value: Value used for padding, defaults to np.nan
        :type pad_value: Union[int, float], optional
        :return: Datacube with second dimension expanded or shortened
        :rtype: np.ndarray
        """        
        diff: int = target - datacube.shape[1]
        if diff < 0:
            datacube = np.delete(datacube, list(range(abs(diff) - 1)), axis=1)  # WAIT A SECOND, '- 1' IS WRONG?
        elif diff > 0:
            datacube = np.pad(datacube, ((0, 0), (diff, 0), (0, 0)), constant_values=pad_value)  # (diff, 0) is start padding instead of end padding; THIS WAS THE ERROR

        assert target == datacube.shape[1]

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
