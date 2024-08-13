time \
    TRITON_NVDISASM_PATH=/usr/local/cuda-11.8/bin/nvdisasm \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda-11.8/bin/cuobjdump \
    TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas \
    inference --weights bert-ii.tar --input-dir /force/FORCE/C1/L2/ard/ --input-tiles tiles.txt --input-glob '*SEN2*BOA.tif' \
    --date-start 20170812 --date-cutoff 20210825 --sequence 256 --batch-size 256 #--mask-dir ./masks/
