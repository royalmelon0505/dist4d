# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import argparse
import subprocess
import multiprocessing as mp
from multiprocessing import Pool, Process
from functools import partial
import os
import torch
import random
# fmt: off
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], './SemSeg/Oneformer/'))

import time
import cv2
import numpy as np
import tqdm
import math
import warnings

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
sys.path.append('./SemSeg/OneFormer/')
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

sys.path.append('./SemSeg/OneFormer/infer')
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
        default="../configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", 
        default="semantic",
        help="Task type")
    parser.add_argument(
        "--input_path",
        help="path to MVS_input path, contain multiple images",
    )
    parser.add_argument(
        "--scale",
        default=0.5, type=float,
        help="resize scale for input images",
    )
    parser.add_argument(
        "--gpu_num",
        default=1, type=int,
        help="gpu number",
    )
    parser.add_argument(
        "--mp_num",
        default=2, type=int,
        help="mp number",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

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

def mp_func(params):
    '''
    多线程加速oneformer infer
    '''
    args, device_name, cam_name, id_s, id_e = params
    
    input_img_path = os.path.join(args.input_path, cam_name, 'images')
    output_path = os.path.join(args.output, cam_name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_name.split(':')[-1]
    cmd = f"python SemSeg/OneFormer/infer/infer_sep.py  \
            --config_file {args.config_file} \
            --task {args.task} \
            --input_path {input_img_path} \
            --scale {args.scale} \
            --start_id {id_s} \
            --end_id {id_e} \
            --output {output_path} \
            --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS SemSeg/OneFormer/ckpt/250_16_dinat_l_oneformer_mapillary_300k.pth"

    try:
        subprocess.run(cmd, shell=True, env=env, check=True)
    except subprocess.CalledProcessError as e:
         print(f"return code: {e.returncode}")


def batch_infer_mp(args):
    print(f"Batch SemSeg using {args.gpu_num} GPUs with {args.mp_num} process per GPU!")
    cam_list = sorted(os.listdir(args.input_path))

    for cam in cam_list:
        img_root = os.path.join(args.input_path, cam, 'images')
        img_list = sorted(os.listdir(img_root))
        inter_num = math.ceil(len(img_list) / (args.gpu_num * args.mp_num))

        params = []
        total_idx = 0
        for gpu_idx in range(args.gpu_num):
            for mp_idx in range(args.mp_num):
                device_name = f"cuda:{gpu_idx}"
                id_s, id_e = total_idx * inter_num, (total_idx + 1) * inter_num
                id_e = len(img_list) if id_e >= len(img_list) else id_e
                params.append([args, device_name, cam, id_s, id_e])
                total_idx += 1

        p_dict ={}
        for p_i in range(args.gpu_num * args.mp_num):
            p_dict[p_i] = Process(target=mp_func, args=[params[p_i]])

        for p_i in range(args.gpu_num * args.mp_num):
            p_dict[p_i].start()

        for p_i in range(args.gpu_num * args.mp_num):
            p_dict[p_i].join()


def mp_main(params):
    '''
    多线程加速oneformer infer
    '''
    args, device_name, cam_name, id_s, id_e, \
        input_path, output, config_file, task, scale = params
    
    input_img_path = os.path.join(input_path, cam_name, 'images')
    output_path = os.path.join(output, cam_name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_name.split(':')[-1]
    cmd = f"python SemSeg/OneFormer/infer/infer_sep.py  \
            --config_file {config_file} \
            --task {task} \
            --input_path {input_img_path} \
            --scale {scale} \
            --start_id {id_s} \
            --end_id {id_e} \
            --output {output_path} \
            --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS SemSeg/OneFormer/ckpt/250_16_dinat_l_oneformer_mapillary_300k.pth"

    try:
        subprocess.run(cmd, shell=True, env=env, check=True)
    except subprocess.CalledProcessError as e:
         print(f"return code: {e.returncode}")


def main(input_path,
        output,
        task,
        scale,
        gpu_num,
        mp_num,
        config_file,
        **kwargs,
        ):
    print(f"Batch SemSeg using {gpu_num} GPUs with {mp_num} process per GPU!")
    # cam_list = sorted(os.listdir(input_path))
    cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
    for cam in cam_list:
        img_root = os.path.join(input_path, cam, 'images')
        img_list = sorted(os.listdir(img_root))
        inter_num = math.ceil(len(img_list) / (gpu_num * mp_num))

        params = []
        total_idx = 0
        args="useless"
        for gpu_idx in range(gpu_num):
            for mp_idx in range(mp_num):
                device_name = f"cuda:{gpu_idx}"
                id_s, id_e = total_idx * inter_num, (total_idx + 1) * inter_num
                id_e = len(img_list) if id_e >= len(img_list) else id_e
                params.append([args, device_name, cam, id_s, id_e, \
                               input_path, output, config_file, task, scale])
                total_idx += 1

        p_dict ={}
        for p_i in range(gpu_num * mp_num):
            p_dict[p_i] = Process(target=mp_main, args=[params[p_i]])

        for p_i in range(gpu_num * mp_num):
            p_dict[p_i].start()

        for p_i in range(gpu_num * mp_num):
            p_dict[p_i].join()


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    args = get_parser().parse_args()
    # mp.set_start_method("spawn", force=True)
    batch_infer_mp(args)