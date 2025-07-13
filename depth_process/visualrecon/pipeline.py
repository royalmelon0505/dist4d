import os, sys, time
from tqdm import tqdm
import torch
from pathlib import Path
from loguru import logger
# import configs.base as cfg
import argparse
import importlib

class pipeline():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def get_cfg(self,):
        return self.cfg
    
    def get_cfg_single(self, cfg_name):
        return getattr(self.cfg, cfg_name, None)
    
    def run(self,):
        action_list = self.cfg.action_list


        if 'semseg' in action_list:
            semseg_t_start = time.time()
            if self.cfg.semseg_cfg['model'] == 'segformer':
                import SemSeg.SemSeg_mc_segformer as semseg_ppl
                logger.info("==> Start semseg using Segformer")
                try:
                    semseg_ppl.main(**self.cfg.semseg_cfg['segformer_cfg'])
                except Exception as e:
                    logger.error(f"==> SemSeg failed! {e}")

            elif self.cfg.semseg_cfg['model'] == 'oneformer':
                import SemSeg.infer_acc_oneformer as oneformer_ppl
                logger.info("==> Start semseg using Oneformer")
                oneformer_ppl.main(**self.cfg.semseg_cfg['oneformer_cfg'])
                # try:
                #     oneformer_ppl.main(**self.cfg.semseg_cfg['oneformer_cfg'])
                # except Exception as e:
                #     logger.error(f"==> SemSeg failed! {e}")

            else:
                raise NotImplementedError
            
            logger.info(f"==> Finish SemSeg using {time.time() - semseg_t_start} s.")
        
        if 'mvs_depth' in action_list:
            import MVS.TransMVSNet_hr.infer as mvs_infer_ppl
            logger.info(f"==> Start MVS_depth inference")
            mvs_depth_t_start = time.time()
            # try:
            mvs_infer_ppl.main(self.cfg.mvs_depth_cfg)
            # except Exception as e:
                # logger.error(f"==> mvs_depth failed! {e}")
            logger.info(f"==> Finish mvs_depth using {time.time() - mvs_depth_t_start} s.")
        
        if 'depth_fuse' in action_list:
            import MVS.TransMVSNet_hr.nus_fusion_acc as depth_fuse_ppl
            logger.info(f"==> Start depth fuse")
            depth_fuse_t_start = time.time()
            try:
                depth_fuse_ppl.main(self.cfg.depth_fuse_cfg)
            except Exception as e:
                logger.error(f"==> depth fuse failed! {e}")
            logger.info(f"==> Finish depth fuse using {time.time() - depth_fuse_t_start} s.")
        
        if 'merge_pcd' in action_list:
            import merge_pcd as merge_pcd_ppl
            logger.info(f"==> Start merge pcd")
            merge_t_start = time.time()
            try:
                merge_pcd_ppl.merge_ply_v3_mp(**self.cfg.merge_pcd_cfg)
            except Exception as e:
                logger.error(f"==> merge pcd failed! {e}")
            logger.info(f"==> Finish merge pcd using {time.time() - merge_t_start} s.")


        if 'clear_MVS_mid' in action_list:
            import MVS.clear_cache as clear_cache
            logger.info(f"==> Start clear mvs cache")
            clear_t_start = time.time()
            try:
                clear_cache.main(**self.cfg.clear_cache_cfg)
            except Exception as e:
                logger.error(f"==> clear cache failed! {e}")
            logger.info(f"==> Finish clear MVS cache using {time.time() - clear_t_start} s.")




def update_configs(indice):
    split_num = 3
    device_cfg = dict(
        gpu_num = 1,
        colmap_exe = f"SfM/colmap_3.8/build/src/exe/colmap"
    )
    
    mvs_input_root = "depth_process/mvs_input_root"
    semseg_cfg = dict(
        model = 'segformer', # ['segformer', 'oneformer']
        segformer_cfg = dict(
            input_root = f"{mvs_input_root}/MVS_Scene{indice}_input",
            save_root_ = f"{mvs_input_root}/MVS_Scene{indice}_input/SemSeg",
            scale = 0.25,
            gpu_num = device_cfg["gpu_num"],
            mp_num= 4,
            palette = 'cityscapes',
            config = "SemSeg/SegFormer/local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py",
            checkpoint = "SemSeg/SegFormer/checkpoints/segformer.b3.1024x1024.city.160k.pth"
        ),
        oneformer_cfg = dict(
            input_path = f"{mvs_input_root}/MVS_Scene{indice}_input",
            output = f"{mvs_input_root}/MVS_Scene{indice}_input/SemSeg_one",
            task = 'semantic',
            scale = 0.25,
            gpu_num = device_cfg["gpu_num"],
            mp_num = 1,
            config_file = "SemSeg/OneFormer/configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml",
            opts = ["MODEL.IS_TRAIN", False, "MODEL.IS_DEMO", True, "MODEL.WEIGHTS", "SemSeg/OneFormer/ckpt/250_16_dinat_l_oneformer_mapillary_300k.pth"]
        ),
    )
    
    mvs_base_cfg = dict(
        testlist = f"MVS/TransMVSNet_hr/lists/nus_li.txt",
        ckpt = f"MVS/TransMVSNet_hr/ckpt/model_bld_hr.ckpt",
        mvs_in = f"/{mvs_input_root}/MVS_Scene{indice}_input",
        mvs_out = f"/{mvs_input_root}/MVS_Scene{indice}_output"
        )

    mvs_depth_cfg = dict(
        dataset = "nus_recon_dataset",
        num_view = 7,
        batch_size=device_cfg['gpu_num']*4,
        interval_scale = 1.0,
        numdepth = 192,
        ndepths = [48, 32, 8],
        depth_inter_r = [4, 1, 0.5],
        cr_base_chs = [8, 8, 8],
        max_h = 900, # 1200,    
        max_w = 1600, # 2000,
        fix_res = False,
        num_worker = 4,
        save_vis = True,
        save_rgb = False,
        testpath = mvs_base_cfg["mvs_in"],
        testlist = mvs_base_cfg["testlist"],
        outdir = mvs_base_cfg["mvs_out"],
        loadckpt = mvs_base_cfg["ckpt"],
    )

    depth_fuse_cfg = dict(
        testpath=mvs_base_cfg['mvs_out'],
        input_path = mvs_base_cfg["mvs_in"],
        testlist = mvs_base_cfg['testlist'],
        outdir = mvs_base_cfg["mvs_out"],
        mp_num = 16,
        scale = 0.8,
        photo_threshold = 0.1,#0.05,
        thres_view = 3,#3, # 4
        Semantic_dilate_size = 0,
        Semantic_path = f"{mvs_input_root}/MVS_Scene{indice}_input/SemSeg",
        save_sem_color = False,
        save_vis = True
    )

    merge_pcd_cfg = dict(
        src_path = mvs_base_cfg["mvs_out"],
        save_path = f"{mvs_base_cfg['mvs_out']}/sp_pcd",
        cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],#['cam_front_120', 'cam_back_120'],
        ds = True,
        ds_cls = 'voxel',
        voxel_size = 0.1,
        split_num = split_num,
        fuse_sem_color = False
    )

    clear_cache_cfg = dict(
        mvs_input_path = mvs_base_cfg["mvs_in"],
        mvs_output_path =  mvs_base_cfg["mvs_out"],
        clear_cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],#['cam_front_120', 'cam_back_120'],
    )
    
    return semseg_cfg,mvs_base_cfg, mvs_depth_cfg,depth_fuse_cfg,merge_pcd_cfg,clear_cache_cfg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visual recon pipline')

    parser.add_argument('--config', type=str, default='configs.base', help='config name')
    parser.add_argument('--indice', type=int, help='the nuscenes indice')
    args = parser.parse_args()
    
    cfg = importlib.import_module(args.config)

    semseg_cfg , mvs_base_cfg, mvs_depth_cfg,depth_fuse_cfg,merge_pcd_cfg,clear_cache_cfg = update_configs(args.indice)
    cfg.semseg_cfg = semseg_cfg
    cfg.mvs_base_cfg = mvs_base_cfg
    cfg.mvs_depth_cfg = mvs_depth_cfg
    cfg.depth_fuse_cfg = depth_fuse_cfg
    cfg.merge_pcd_cfg = merge_pcd_cfg
    cfg.clear_cache_cfg = clear_cache_cfg

    ppl = pipeline(cfg=cfg)
    torch.multiprocessing.set_start_method('spawn')
    ppl.run()