export task=semantic

# python demo.py --config-file ../configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml \
#   --input /data/Offline_recon/Car9-recon/data/proc_ppl_bag_20230805_115745_det/MVS_input_fixBug_Sys/cam_front_120/images/00000565.jpg \
#   --output /data/Offline_recon/Car9-recon/data/proc_ppl_bag_20230805_115745_det/OneformerSeg/cam_front_120/00000565.png \
#   --task $task \
#   --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ../ckpt/250_16_dinat_l_oneformer_mapillary_300k.pth

python infer_acc.py \
  --task semantic \
  --scale 0.25 \
  --gpu_num 1 \
  --mp_num 1 \
  --config_file ../configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml \
  --input_path /data/Offline_recon/Car9-recon/data/proc_ppl_bag_20230805_115745_det/MVS_input_fixBug_Sys \
  --output /data/Offline_recon/Car9-recon/data/proc_ppl_bag_20230805_115745_det/SemSeg_OneFromer_test \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ../ckpt/250_16_dinat_l_oneformer_mapillary_300k.pth

# python infer_acc.py \
  # --config-file ../configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml \
  # --input_path /data/Offline_recon/Car9-recon/data/proc_ppl_bag_20230805_115745_det/MVS_input_fixBug_Sys \
  # --output /data/Offline_recon/Car9-recon/data/proc_ppl_bag_20230805_115745_det/SemSeg_OneFromer \
  # --task semantic \
  # --scale 0.4 \
  # --gpu_num 1 \
  # --mp_num 1 \
  # --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ../ckpt/250_16_swin_l_oneformer_mapillary_300k.pth