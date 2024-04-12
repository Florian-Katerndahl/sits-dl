# SITS DL — Python Package to Facilitate Inference of Different Deep Learning Models

This Python package combines different deep learning models, namely one LSTM and two different transformer architectures to allow efficient code re-usage.

## Installation

This packages uses Poetry for dependency management, isolation and packaging. Depending on how you want to install/work with `sits_dl`, you may need to install [Poetry](https://python-poetry.org/) priot.

To install the package for testing/prototyping purposes, simply clone this repo and run `poetry install`. Note that by default, the CUDA 12 wheels of Pytorch are installed. If you want to use other versions, refer to the [Poetry documentation](https://python-poetry.org/docs/cli#source) on how to change package sources.

## Applications Distributed with This Package

Installing the package also gives acces to the application `inference` which can be used from anywhere on your command line after installing. Though note, that when only using `poetry install`, you also need to activate the poetry shell before the application is available.

```
~$ inference --help

usage: inference [-h] -w WEIGHTS --input-tiles INPUT [--input-dir BASE] [--input-glob IGLOB] [--output-dir OUT] [--date-start DATE_BEGIN] --date-cutoff DATE_END 
                 [--mask-dir MASKS] [--mask-glob MGLOB] [--row-size ROW-BLOCK] [--col-size COL-BLOCK] [--log] [--log-file LOG-FILE] [--cpus CPUS] [--batch-size BATCH-SIZE] [--gpu GPU] [--multi-gpu] [--sequence SEQUENCE]

Run inference with already trained LSTM classifier on a remote-sensing time series represented as FORCE ARD datacube.

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS, --weights WEIGHTS
                        Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or absolute file path.
  --input-tiles INPUT   List of FORCE tiles which should be used for inference. Each line should contain one FORCE tile specifier (Xdddd_Ydddd).
  --input-dir BASE      Path to FORCE datacube. By default, use the current PWD.
  --input-glob IGLOB    Optional glob pattern to restricted files used from `input-dir`.
  --output-dir OUT      Path to directory into which predictions should be saved. By default, use the current PWD.
  --date-start DATE_BEGIN
                        Begin date for time series which should be included in datacue for inference. If not specified, take all images up to `date-cutoff`.
  --date-cutoff DATE_END
                        Cutoff date for time series which should be included in datacube for inference.
  --mask-dir MASKS      Path to directory containing folders in FORCE tile structure storing binary masks with a value of 1 representing pixels to predict. Others can be nodata or 0. Masking is done on a row-by-
                        row basis. I.e., the entire unmasked datacube is constructed from the files found in `input-dir`. Only when handing a row of pixels to the DL-model for inference are data removed. Thus,
                        this does not reduce the memory footprint, but can speed up inference significantly under certain conditions.
  --mask-glob MGLOB     Optional glob pattern to restricted file used from `mask-dir`.
  --row-size ROW-BLOCK  Row-wise size to read in at once. If not specified, query dataset for block size and assume constant block sizes across all raster bands in case of multilayer files. Contrary to what GDAL
                        allows, if the entire raster extent is not evenly divisible by the block size, an error will be raised and the process aborted. If only `row-size` is given, read the specified amount of
                        rows and however many columns are given by the datasets block size. If both `row-size` and `col-size` are given, read tiles of specified size.
  --col-size COL-BLOCK  Column-wise size to read in at once. If not specified, query dataset for block size and assume constant block sizes across all raster bands in case of multilayer files. Contrary to what
                        GDAL allows, if the entire raster extent is not evenly divisible by the block size, an error will be raised and the process aborted. If only `col-size` is given, read the specified amount
                        of columns and however many rows are given by the datasets block size. If both `col-size` and `row-size` are given, read tiles of specified size.
  --log                 Emit logs?
  --log-file LOG-FILE   If logging is enabled, write to this file. If omitted, logs are written to stdout.
  --cpus CPUS           Number of CPUS for Inter-OP and Intra-OP parallelization of pytorch.
  --batch-size BATCH-SIZE
                        Batch size used during inference when using Transformer DL models. Default is 1024.
  --gpu GPU             Select a GPU. Default is 'cuda:0'.
  --multi-gpu           Enable multi-gpu usage. Default is False. Only applicable for transformer models.
  --sequence SEQUENCE   Transformer sequence length. Only needed when using transformer model for inference.
```
