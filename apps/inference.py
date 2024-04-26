import argparse
from datetime import datetime
from pathlib import Path
from re import search
from typing import List, Union, Dict, Optional
import numpy as np
import rasterio
import rioxarray as rxr
import torch
from torch.nn import DataParallel
import torch.multiprocessing as mp
import xarray
import logging
from time import time
from sits_dl.tensordatacube import TensorDataCube, Models
from sits_dl.sbert import SBERT, SBERTClassification
from sits_dl import lstm
from sits_dl import transformer


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run inference with already trained LSTM classifier on a remote-sensing time series represented as "
        "FORCE ARD datacube."
    )
    parser.add_argument(
        "-w",
        "--weights",
        dest="weights",
        required=True,
        type=Path,
        help="Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or "
        "absolute file path.",
    )
    parser.add_argument(
        "--input-tiles",
        dest="input",
        required=True,
        type=Path,
        help="List of FORCE tiles which should be used for inference. Each line should contain one FORCE "
        "tile specifier (Xdddd_Ydddd).",
    )
    parser.add_argument(
        "--input-dir",
        dest="base",
        required=False,
        type=Path,
        default=Path("."),
        help="Path to FORCE datacube. By default, use the current PWD.",
    )
    parser.add_argument(
        "--input-glob",
        dest="iglob",
        required=False,
        type=str,
        default="*",
        help="Optional glob pattern to restricted files used from `input-dir`.",
    )
    parser.add_argument(
        "--output-dir",
        dest="out",
        required=False,
        type=Path,
        default=Path("."),
        help="Path to directory into which predictions should be saved. By default, use the " "current PWD.",
    )
    parser.add_argument(
        "--date-start",
        dest="date_begin",
        required=False,
        type=int,
        default=None,
        help="Begin date for time series which should be included in datacue for inference. "
        "If not specified, take all images up to `date-cutoff`."
    )
    parser.add_argument(
        "--date-cutoff",
        dest="date_end",
        required=True,
        type=int,
        help="Cutoff date for time series which should be included in datacube for inference.",
    )
    parser.add_argument(
        "--mask-dir",
        dest="masks",
        required=False,
        type=Path,
        default=None,
        help="Path to directory containing folders in FORCE tile structure storing "
        "binary masks with a value of 1 representing pixels to predict. Others can be nodata "
        "or 0. Masking is done on a row-by-row basis. I.e., the entire unmasked datacube "
        "is constructed from the files found in `input-dir`. Only when handing a row of "
        "pixels to the DL-model for inference are data removed. Thus, this does not reduce "
        "the memory footprint, but can speed up inference significantly under certain "
        "conditions.",
    )
    parser.add_argument(
        "--mask-glob",
        dest="mglob",
        required=False,
        type=str,
        default="mask.tif",
        help="Optional glob pattern to restricted file used from `mask-dir`.",
    )
    parser.add_argument(
        "--row-size",
        dest="row-block",
        required=False,
        type=int,
        default=None,
        help="Row-wise size to read in at once. If not specified, query dataset for block size and assume "
        "constant block sizes across all raster bands in case of multilayer files. Contrary to "
        "what GDAL allows, if the entire raster extent is not evenly divisible by the block size, "
        "an error will be raised and the process aborted. If only `row-size` is given, read the "
        "specified amount of rows and however many columns are given by the datasets block size. "
        "If both `row-size` and `col-size` are given, read tiles of specified size.",
    )
    parser.add_argument(
        "--col-size",
        dest="col-block",
        required=False,
        type=int,
        default=None,
        help="Column-wise size to read in at once. If not specified, query dataset for block size and "
        "assume constant block sizes across all raster bands in case of multilayer files. Contrary to "
        "what GDAL allows, if the entire raster extent is not evenly divisible by the block size, "
        "an error will be raised and the process aborted. If only `col-size` is given, read the "
        "specified amount of columns and however many rows are given by the datasets block size. "
        "If both `col-size` and `row-size` are given, read tiles of specified size.",
    )
    parser.add_argument("--log", dest="log", required=False, action="store_true", help="Emit logs?")
    parser.add_argument(
        "--log-file",
        dest="log-file",
        required=False,
        type=str,
        help="If logging is enabled, write to this file. If omitted, logs are written to stdout.",
    )
    parser.add_argument(
        "--cpus",
        dest="cpus",
        required=False,
        default=None,
        type=int,
        help="Number of CPUS for Inter-OP and Intra-OP parallelization of pytorch.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch-size",
        required=False,
        type=int,
        default=1024,
        help="Batch size used during inference when using Transformer DL models. Default is 1024.",
    )
    parser.add_argument(
        "--gpu", dest="gpu", required=False, type=str, default="cuda:0", help="Select a GPU. Default is 'cuda:0'."
    )
    parser.add_argument(
        "--multi-gpu", dest="mgpu", required=False, action="store_true",
        help="Enable multi-gpu usage. Default is False. Only applicable for transformer models."
    )
    parser.add_argument(
        "--sequence",
        dest="sequence",
        required=False,
        type=int,
        default=None,
        help="Transformer sequence length. Only needed when using transformer model for inference.",
    )

    cli_args: Dict[str, Union[Path, int, bool, str]] = vars(parser.parse_args())

    # TODO look into how the logging module really works and insert logging statements in other parts of
    #      the lib?
    if cli_args.get("log"):
        if cli_args.get("log-file"):
            logging.basicConfig(level=logging.INFO, filename=cli_args.get("log-file"))
        else:
            logging.basicConfig(level=logging.INFO)

    if cli_args.get("cpus"):
        torch.set_num_threads(cli_args.get("cpus"))
        torch.set_num_interop_threads(cli_args.get("cpus"))

    device: str = torch.device(cli_args.get("gpu") if torch.cuda.is_available() else "cpu")

    # FIXME this is somewhat ugly
    inference_model: Union[lstm.LSTMClassifier, transformer.TransformerClassifier, SBERTClassification]
    inference_type: Models
    try:
        inference_model = torch.load(
        cli_args.get("weights"), map_location=device
        ).eval()
        if "lstm" in inference_model.__module__.lower():
            inference_type = Models.LSTM
            inference_model.predict = lstm.LSTMClassifier.predict.__get__(inference_model)  # https://stackoverflow.com/a/2982; https://stackoverflow.com/a/73581487
        elif "transformer" in inference_model.__module__.lower():
            inference_type = Models.TRANSFORMER
            inference_model.predict = transformer.TransformerClassifier.predict.__get__(inference_model)  # https://stackoverflow.com/a/2982; https://stackoverflow.com/a/73581487
        else:
            raise RuntimeError("Unknown model type supplied")
    except AttributeError:
        SBERTArgs: Dict = {'num_features': 10, 'hidden': 128, 'n_layers': 3, 'attn_heads': 8, 'dropout': 0.3}
        SBERTClassArgs: Dict = {'num_classes': 1, 'seq_len': cli_args.get("sequence")}
        
        checkpoint = torch.load(cli_args.get("weights"), map_location=device)
        inference_type = Models.SBERT
        inference_model = SBERTClassification(SBERT(**SBERTArgs), **SBERTClassArgs)
        inference_model.load_state_dict(checkpoint["model_state_dict"])
        inference_model.to(device)
        inference_model.eval()

        if cli_args.get("mgpu"):
            inference_model = DataParallel(inference_model)

    if (inference_type == Models.SBERT or inference_type == Models.TRANSFORMER) and not cli_args.get("sequence"):
        raise RuntimeError("Provided a transformer model without providing the sequence length."
                           "Please re-run with `--sequence` set.")

    with open(cli_args.get("input"), "rt") as f:
        force_tiles: List[str] = [tile.replace("\n", "") for tile in f.readlines()]

    mp.set_sharing_strategy("file_system")

    for tile in force_tiles:
        tdc: TensorDataCube = TensorDataCube(
            cli_args.get("base"),
            tile,
            cli_args.get("iglob"),
            cli_args.get("date_begin"),
            cli_args.get("date_end"),
            inference_type,
            cli_args.get("row-block"),
            cli_args.get("col-block"),
            cli_args.get("sequence"),
        )
        tdc.self_prep()
        output_torch: torch.Tensor = tdc.empty_output(torch.float if inference_type == Models.SBERT else torch.long)

        for chunk, true_obs, row, col in tdc:
            r_start, r_end, c_start, c_end = row, row + tdc.row_step, col, col + tdc.column_step
            mask: Optional[np.ndarray] = None

            if cli_args.get("masks"):
                mask_dir: Path = cli_args.get("masks") / tile
                if not mask_dir.exists():
                    mask_dir = mask_dir.parent

                try:
                    mask_path: str = str(list(mask_dir.glob(cli_args.get("mglob"))).pop())
                except IndexError:
                    raise RuntimeError("Provided mask directory and name but could not find any files")

                with rxr.open_rasterio(mask_path) as ds:
                    mask_ds: xarray.Dataset = ds.isel(
                        y=slice(row, row + tdc.row_step), x=slice(col, col + tdc.column_step)
                    )
                    mask = np.squeeze(np.array(mask_ds, ndmin=2, dtype=np.bool_), axis=0)
                    del mask_ds

            output_torch[r_start:r_end, c_start:c_end] = inference_model.predict(
                chunk,
                mask,
                tdc.column_step,
                tdc.row_step,
                cli_args.get("batch-size"),
                device
            )

            del chunk

        output_numpy: np.array = output_torch.numpy(force=True)

        with rasterio.open(cli_args.get("out") / (tile + ".tif"), "w", **tdc.output_metadata) as dst:
            dst.write_band(1, output_numpy.astype(rasterio.float32 if inference_type == Models.SBERT else rasterio.uint8))

        del output_torch
        del output_numpy
    
    return 0
