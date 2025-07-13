
GPUS=2
CFG=./configs/magicdrive/test/infer_dataset_424full.py

PATH_TO_MODEL=./ckpt/outputs_424_onlyRGB/ema.pt
FRAME=full
torchrun --standalone --nproc_per_node ${GPUS} scripts/infer_dist_dataset_full_onlyRGB.py ${CFG} \
    --cfg-options model.from_pretrained=${PATH_TO_MODEL}
