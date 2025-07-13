'''
multi-cam multi-process verision to perform semseg.
'''
import os, sys
import math
import time
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
# sys.path.append(str(Path(__file__).parent))
from SemSeg.SegFormer.mmseg.apis import inference_segmentor, init_segmentor, save_results_all
from SemSeg.SegFormer.mmseg.core.evaluation import get_palette
        

def batch_demo(img_root, save_root, config, checkpoint, device='cuda:0', palette='cityscapes'):
    '''
    '''
    assert os.path.isdir(img_root), f"{img_root} doesn't exist!"
    os.makedirs(save_root, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, checkpoint, device=device)

    img_subfolder = img_root
    save_subfolder = save_root
    os.makedirs(save_subfolder, exist_ok=True)
    os.makedirs(os.path.join(save_subfolder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_subfolder, 'visualize'), exist_ok=True)
    os.makedirs(os.path.join(save_subfolder, 'color_seg'), exist_ok=True)
    os.makedirs(os.path.join(save_subfolder, 'seg'), exist_ok=True)
    img_list = sorted(os.listdir(img_subfolder))

    t0 = time.time()
    for img in tqdm(img_list, desc=f"semanSeg"):
        img_fullname = os.path.join(img_subfolder, img)
        result = inference_segmentor(model, img_fullname)

        # save the single result
        save_results_all(model, img_fullname, result, get_palette(palette),
                    save_path=save_subfolder, result_name=img)
    print(f"Time cost = {time.time()-t0} s.")


def mp(params):
    img_root, save_root, scale, id_s, id_e, config, checkpoint, device, palette, cam_name = params

    model = init_segmentor(config, checkpoint, device=device)
    img_list = sorted(os.listdir(img_root))

    t0 = time.time()
    for img in tqdm(img_list[id_s:id_e], desc=f"SemSeg-{cam_name}"):
        img_fullname = os.path.join(img_root, img)
        result = inference_segmentor(model, img_fullname, scale)

        # save the single result
        save_results_all(model, img_fullname, result, get_palette(palette),
                    save_path=save_root, result_name=img)
    print(f"\n ({device}) Segment image-id {id_s}-{id_e-1} cost time = {time.time()-t0} s.")


def batch_demo_mp(img_root, save_root, config, checkpoint, scale, gpu_num=1, mp_num=2, palette='cityscapes'):
    
    print(f"Batch SemSeg using {gpu_num} GPUs with {mp_num} process per GPU!")

    assert os.path.isdir(img_root), f"{img_root} doesn't exist!"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'visualize'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'color_seg'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'seg'), exist_ok=True)

    img_list = sorted(os.listdir(img_root))
    inter_num = math.ceil(len(img_list) / (gpu_num * mp_num))
    params = []
    total_idx = 0
    for gpu_idx in range(gpu_num):
        for mp_idx in range(mp_num):
            device = f"cuda:{gpu_idx}"
            id_s, id_e = total_idx * inter_num, (total_idx + 1) * inter_num
            id_e = len(img_list) if id_e >= len(img_list) else id_e
            params.append([img_root, save_root, id_s, id_e, config, \
                           checkpoint, scale, device, palette])
            total_idx += 1

    partial_func = partial(mp)
    p = Pool(gpu_num*mp_num)
    p.map(partial_func, params)
    p.close()
    p.join()


def main(input_root, save_root_, config, checkpoint, scale, gpu_num=1, mp_num=2, palette='cityscapes'):
    
    print(f"Batch SemSeg using {gpu_num} GPUs with {mp_num} process per GPU!")
    cam_list = sorted(os.listdir(input_root))

    for cam in cam_list:
        img_root = os.path.join(input_root, cam, 'images')
        save_root = os.path.join(save_root_, cam)

        assert os.path.isdir(img_root), f"{img_root} doesn't exist!"
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(os.path.join(save_root, 'mask'), exist_ok=True)
        # os.makedirs(os.path.join(save_root, 'visualize'), exist_ok=True)
        # os.makedirs(os.path.join(save_root, 'color_seg'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'seg'), exist_ok=True)

        img_list = sorted(os.listdir(img_root))
        inter_num = math.ceil(len(img_list) / (gpu_num * mp_num))
        
        params = []
        total_idx = 0
        for gpu_idx in range(gpu_num):
            for mp_idx in range(mp_num):
                device = f"cuda:{gpu_idx}"
                id_s, id_e = total_idx * inter_num, (total_idx + 1) * inter_num
                id_e = len(img_list) if id_e >= len(img_list) else id_e
                params.append([img_root, save_root, scale, id_s, id_e, \
                                config, checkpoint, device, palette, cam])
                total_idx += 1

        partial_func = partial(mp)
        p = Pool(gpu_num * mp_num)
        p.map(partial_func, params)
        p.close()
        p.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Batch semseg images')

    parser.add_argument('--input_root',
                        type=str,
                        required=True,
                        help='root path to mvs input (contain multi-cams).')
    parser.add_argument('--save_root',
                        type=str,
                        required=True,
                        help='root path to save semantic results.')
    parser.add_argument('--resize_scale',
                        type=float,
                        default=1.0,
                        help='resize scale of the input image.')
    parser.add_argument('--config_file',
                        type=str,
                        default='./SemSeg/SegFormer/local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py',
                        help='path to config file.')
    parser.add_argument('--ckpt_file',
                        type=str,
                        default='./SemSeg/SegFormer/checkpoints/segformer.b3.1024x1024.city.160k.pth',
                        help='path to checkpoint file.')
    parser.add_argument('--gpu_num',
                        type=int,
                        default=1,
                        help='gpu numbers to suport SemSeg.')
    parser.add_argument('--mp_num',
                        type=int,
                        default=1,
                        help='multi-process numbers, max=2 for single 2080Ti.')

    args = parser.parse_args()

    config_file = args.config_file
    ckpt_file = args.ckpt_file
    input_root = args.input_root
    save_root = args.save_root

    # torch.multiprocessing.set_start_method('spawn')
    main(input_root, save_root, config_file, ckpt_file, 
                     args.resize_scale, args.gpu_num, args.mp_num)
