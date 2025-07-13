GPUS=8

CFG=./configs/magicdrive/train/train_9-17x424x800only_stdit3-mm_onlyRGB.py
torchrun --nproc-per-node ${GPUS}  --nnode=1 \
    scripts/train_dist_mm_onlyRGBcondition.py ${CFG} --cfg-options num_workers=6 prefetch_factor=3
