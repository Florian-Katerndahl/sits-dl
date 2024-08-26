from datetime import datetime
from time import time
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from re import search, Pattern, compile
import multiprocessing as mp
from warnings import warn
import torch
from torch.utils.data import DataLoader
import rasterio
import numpy as np
from sits_dl.forestmask import ForestMask


@torch.compile(dynamic=False, fullgraph=True)
def znorm_psb_t(dc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-normalize input data cube in PSB form (pixels, sequence, spectral bands) across sequence

    .. note:: This function is `torch.compile`'ed!

    :param dc: Data cube to normalize
    :type dc: torch.Tensor
    :param eps: Epsilon, defaults to 1e-6
    :type eps: float, optional
    :return: Z-normalized tensor
    :rtype: torch.Tensor
    """    
    _mean: torch.Tensor = torch.nanmean(dc, dim=1, keepdim=True)
    _std = (dc - _mean).square().nanmean(dim=1, keepdim=True).sqrt()
    normed = (dc - _mean) / _std + eps
    return normed


@torch.compile()
def preprocess_sbert(dc: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process SBERT data cube. I.e., move all valid observations to the "front" of the data
      cube and calculate a mask for all valid/invalid observations

    .. note:: This function is `torch.compile`'ed!

    :param dc: Data cube to process with [Sequence, Bands, X, Y]
    :type dc: np.ndarray
    :param device: torch device on which data is to be processed
    :type device: torch.device
    :return: processed tensor, tensor used to sort processed tensor and mask tensor
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """    
    dc_t: torch.Tensor = (
        torch.from_numpy(dc).to(device=device).permute(2, 3, 0, 1).reshape((-1, *dc.shape[:2]))
    )
    dc_t[dc_t == -9999] = torch.nan
    pixels, sequence_length, bands = dc_t.shape
    observations_without_nodata: torch.Tensor = ~dc_t.isnan().any(dim=2).to(device=device)
    dc_t[~observations_without_nodata] = torch.nan
    index_array: torch.Tensor = (
        torch.arange(sequence_length)
        .reshape((1, sequence_length, 1))
        .repeat_interleave(pixels, dim=0)
        .to(device=device)
    )
    index_array[observations_without_nodata] -= sequence_length * 2
    sorting_keys: torch.Tensor = index_array.argsort(dim=1)
    dc_t = dc_t.take_along_dim(sorting_keys, dim=1)

    sbert_mask = ~dc_t.isnan().any(dim=2).reshape((pixels, sequence_length, 1))

    del pixels, sequence_length, bands, index_array

    return dc_t, sorting_keys, sbert_mask


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
        chunk_queue: mp.Queue = None,
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
        self.chunk_queue = chunk_queue

        self.cube_inputs: Optional[List[str]] = None
        self.image_info: Optional[Dict] = None
        self.output_metadata: Optional[Dict] = None

    def self_prep(self) -> None:
        """Gather input files, filter them by date and acquire image properties

        This method is to be called before actual processing stars.

        :raises AssertionError: If rows and/or columns of input images are not equally divisible by chosen
          step size.
        """        
        warn(
            (
                "Only front padding implemented currently. I.e. If DC is your datacube and n < seq, the first n-seq "
                "items are filled with NAs and only the last seq-many items are valid observations"
            )
        )
        tile_paths: List[str] = [
            str(p) for p in (self.input_directory / self.tile_id).glob(self.input_glob)
        ]
        if self.start:
            self.cube_inputs = [
                tile_path
                for tile_path in tile_paths
                if self.start <= int(search(r"\d{8}", tile_path).group(0)) <= self.cutoff
            ]
        else:
            self.cube_inputs = [
                tile_path
                for tile_path in tile_paths
                if int(search(r"\d{8}", tile_path).group(0)) <= self.cutoff
            ]
        self.cube_inputs.sort()

        with rasterio.open(self.cube_inputs[0]) as f:
            self.output_metadata = f.meta
            input_bands, self.output_metadata["count"] = self.output_metadata["count"], 1
            self.output_metadata["dtype"] = (
                rasterio.float32 if self.inference_type == Models.SBERT else rasterio.uint8
            )
            self.output_metadata["nodata"] = TensorDataCube.OUTPUT_NODATA
            row_block, col_block = f.block_shapes[0]

        self.image_info = {
            "input_bands": input_bands,
            "tile_width": self.output_metadata["width"],
            "tile_height": self.output_metadata["height"],
        }

        self.row_step = self.row_step or row_block
        self.column_step = self.column_step or col_block

        if (self.image_info["tile_width"] % self.column_step) != 0 or (
            self.image_info["tile_height"] % self.row_step
        ) != 0:
            raise AssertionError(
                "Rows and columns must be divisible by their respective step sizes without remainder."
            )

    def __iter__(self):
        """Process chunks of data cube in iterative fashion

        :yield: Tensor with normalized spectral values, DOY and mask, forest mask tensor, etc
        :rtype: Tuple
        """        
        for row in range(0, self.image_info["tile_height"], self.row_step):
            for col in range(0, self.image_info["tile_width"], self.column_step):
                t_chunk = time()
                s2_cube_np, mask, index_offset = self.chunk_queue.get()

                if self.inference_type == Models.TRANSFORMER:
                    _t: Tuple[
                        List[Union[datetime, float]], List[bool]
                    ] = TensorDataCube.pad_doy_sequence(
                        self.sequence_length, self.cube_inputs, self.inference_type
                    )
                    sensing_doys, sbert_mask = _t
                    sensing_doys_np: np.ndarray = np.array(sensing_doys)
                    sensing_doys_np = sensing_doys_np.reshape((self.sequence_length, 1, 1, 1))
                    sensing_doys_np = np.repeat(
                        sensing_doys_np, self.row_step, axis=2
                    )  # match actual data cube
                    sensing_doys_np = np.repeat(
                        sensing_doys_np, self.column_step, axis=3
                    )  # match actual data cube

                    # lines below needed now for compatability with SBERT datacube
                    s2_cube_np = s2_cube_np.transpose((2, 3, 0, 1))
                    s2_cube_np = s2_cube_np.reshape((-1, *s2_cube_np.shape[2:]))
                    sensing_doys_np = sensing_doys_np.transpose((2, 3, 0, 1))
                    sensing_doys_np = sensing_doys_np.reshape((-1, *sensing_doys_np.shape[2:]))
                elif self.inference_type == Models.SBERT:
                    s2_cube_np, sorting_keys, sbert_mask = preprocess_sbert(s2_cube_np, self.device)
                    pixels, sequence_length, _ = s2_cube_np.shape

                    sensing_dates_as_ordinal: np.ndarray = np.zeros(
                        (sequence_length,), dtype=np.int32
                    )
                    sensing_dates_as_ordinal[index_offset:] = [
                        TensorDataCube.ordinal_observation(i) for i in self.cube_inputs
                    ]
                    sensing_doys_np = sensing_dates_as_ordinal.reshape(
                        (1, sequence_length, 1)
                    ).repeat(pixels, axis=0)
                    sensing_doys_np = np.take_along_axis(
                        sensing_doys_np, sorting_keys.numpy(force=True), axis=1
                    )
                    doy_of_earliest_observations = (
                        TensorDataCube.toyday(sensing_doys_np[:, 0, 0])
                        .reshape((pixels, 1))
                        .repeat(sequence_length, axis=1)
                    )
                    sensing_doys_np = (
                        sensing_doys_np[..., 0]
                        - sensing_doys_np[:, 0].repeat(sequence_length, axis=1)
                        + doy_of_earliest_observations
                    ).reshape((pixels, sequence_length, 1))
                    sensing_doys_np[~sbert_mask.numpy(force=True)] = 0

                if self.inference_type in [Models.TRANSFORMER, Models.SBERT]:
                    s2_cube_np = znorm_psb_t(s2_cube_np)
                    s2_cube_np[s2_cube_np.isnan()] = 0.0

                if self.inference_type == Models.SBERT:
                    cube_and_doys = torch.cat(
                        [
                            s2_cube_np,
                            torch.from_numpy(sensing_doys_np).to(device=self.device),
                            sbert_mask,
                        ],
                        dim=2,
                    )
                    del (
                        sbert_mask,
                        sensing_dates_as_ordinal,
                        sensing_doys_np,
                        doy_of_earliest_observations,
                        sorting_keys,
                    )
                elif self.inference_type == Models.TRANSFORMER:
                    cube_and_doys = torch.cat(
                        [s2_cube_np, torch.from_numpy(sensing_doys_np).to(device=self.device)],
                        dim=2,
                    )
                    del sensing_doys_no, sbert_mask, sensing_doys
                elif self.inference_type == Models.LSTM:
                    cube_and_doys = s2_cube_np

                yield cube_and_doys, mask, row, col, t_chunk

    def empty_output(self, dtype: torch.dtype = torch.long) -> torch.Tensor:
        """Create output tensor into which predictions can be stored, filled with output nodata value

        :param dtype: Data type of returned tensor, defaults to torch.long
        :type dtype: torch.dtype, optional
        :return: Filled output tensor
        :rtype: torch.Tensor
        """        
        return torch.full(
            [self.image_info["tile_height"], self.image_info["tile_width"]],
            fill_value=TensorDataCube.OUTPUT_NODATA,
            dtype=dtype,
            device="cpu",
        )

    @classmethod
    def to_dataloader(cls, chunk: torch.Tensor, batch_size: int, workers: int = 4) -> DataLoader:
        return DataLoader(
            chunk,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
            persistent_workers=False,
        )

    @classmethod
    def pad_doy_sequence(
        cls, target: int, observations: List[str], mtype: Models
    ) -> Tuple[List[Union[datetime, float]], List[bool]]:
        diff: int = target - len(observations)
        true_observations: List[bool] = [True] * len(observations)
        if diff < 0:
            observations = observations[abs(diff) :]  # deletes oldest entries first
            true_observations = true_observations[abs(diff) :]
        elif diff > 0:
            observations += [0.0] * diff
            true_observations += [False] * diff

        # TODO remove assertion for "production"
        assert target == len(observations)

        return [
            TensorDataCube.fp_to_doy(i) for i in observations if isinstance(i, str)
        ], true_observations

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
    def toyday(x: np.ndarray) -> np.ndarray:
        """Calculate day of year of numpy array in vectorized form

        :param x: Numpy array
        :type x: np.ndarray
        :return: Numpy array with DOY's
        :rtype: np.ndarray
        """        
        return datetime.fromordinal(x).timetuple().tm_yday

    @staticmethod
    def ordinal_observation(x: str) -> int:
        """Compute ordinal date from input file names

        .. warning:: This method assumes, that there is indeed a correctly formatted
          date sub-string in x.

        :param x: String (file name) containing date string in 'YYYYMMDD' format
        :type x: str
        :return: Date in ordinal form
        :rtype: int
        """        
        return datetime.strptime(TensorDataCube.DATE_IN_FPATH.findall(x)[0], "%Y%m%d").toordinal()
