# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import argparse
import multiprocessing as mp
from multiprocessing import Pool, Process
from functools import partial
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import cv2
import numpy as np
from tqdm import tqdm
import math
import warnings

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

import json
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo, save_pfm

# constants
WINDOW_NAME = "OneFormer Demo"
warnings.filterwarnings("ignore")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config_file",
        default="SemSeg/OneFormer/configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", 
        default="semantic",
        help="Task type")
    parser.add_argument("--device_name", 
        default="cuda:0",
        help="GPU device name")
    parser.add_argument(
        "--scene_root",
        help="path to MVS_input/cam_name/images path, contain only images",
    )
    parser.add_argument(
        "--scale",
        default=0.5, type=float,
        help="resize scale for input images",
    )
    parser.add_argument(
        "--start_id",
        default=0, type=int,
    )
    parser.add_argument(
        "--end_id",
        default=-1, type=int,
    )
    parser.add_argument(
        "--cam_name",
        type=str,
    )
    # parser.add_argument(
    #     "--output",
    #     help="A file or directory to save output visualizations. "
    #     "If not given, will show output in an OpenCV window.",
    # )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def infer(args):
    # print(args)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    with open(os.path.join(args.scene_root,"sample_token_list.json"),'r') as file:
        sample_token_list = json.load(file)

    # src_img_path = args.input_path
    # img_list = sorted(os.listdir(src_img_path))
    # inputs_list = [os.path.join(src_img_path, item) for item in img_list]

    # save_vis_path = os.path.join(args.output, 'color_seg')
    # save_raw_path = os.path.join(args.output, 'seg')
    # if not os.path.isdir(save_vis_path) or not os.path.isdir(save_raw_path):
    #     os.makedirs(save_vis_path, exist_ok=True)
    #     os.makedirs(save_raw_path, exist_ok=True)
    idx_list = list(range(args.start_id,args.end_id))
    for idx in tqdm(idx_list, desc=f"SemSeg-{args.scene_root[-10:]}-{args.cam_name}"):
        sample_token = sample_token_list[idx]
        input_npz_path = os.path.join(args.scene_root,sample_token,args.cam_name,f"rgb_depth_{idx}.npz")
        rgb_depth_data = np.load(input_npz_path)
        input_image = rgb_depth_data['rgb'] #nparray

        input_image = input_image[:, :, ::-1]  # RGB 2 BGR
        h, w, _ = input_image.shape
        img = cv2.resize(input_image, (int(w*args.scale), int(h*args.scale)))
        with torch.no_grad():
            predictions, visualized_output = demo.run_on_image(img, args.task)

        Seg_result = predictions['sem_seg'].argmax(dim=0).cpu().numpy()
        save_sem_npz_path = os.path.join(args.scene_root,sample_token,args.cam_name,f"semantic_oneformer.npz")
        np.savez_compressed(save_sem_npz_path,sem=Seg_result)

        # visualized_output['semantic_inference'].save(os.path.join(args.scene_root,sample_token,args.cam_name,f"vis_semantic_oneformer.jpg"))


    # for path in tqdm.tqdm(inputs_list[args.start_id:args.end_id], desc=f"run oneformer for {args.start_id}-{args.end_id} "):
    #     img = read_image(path, format="BGR")    # [h,w,3]
    #     h, w, _ = img.shape
    #     img = cv2.resize(img, (int(w*args.scale), int(h*args.scale)))
    #     with torch.no_grad():
    #         predictions, visualized_output = demo.run_on_image(img, args.task)
    #     # predictions['sem_seg'].shape = tensor,[N_cls,h,w]
    #     out_vis_filename = os.path.join(save_vis_path, os.path.basename(path))
    #     visualized_output['semantic_inference'].save(out_vis_filename)

    #     out_raw_filename = os.path.join(save_raw_path, f"{os.path.basename(path).split('.')[0]}.pfm")
    #     out_raw = predictions['sem_seg'].argmax(dim=0).cpu().numpy().astype(np.float32)
    #     save_pfm(out_raw_filename, out_raw)


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    args = get_parser().parse_args()
    mp.set_start_method("spawn", force=True)
    infer(args)