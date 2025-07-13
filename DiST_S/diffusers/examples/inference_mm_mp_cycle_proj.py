import sys
import torch
import os
# from src.pipelines.pipeline_stable_video_diffusion_custom import StableVideoDiffusionPipeline_convnb_multiframe
from src.pipelines.pipeline_stable_video_diffusion_custom_mm import StableVideoDiffusionPipeline_convnb_multiframe_mm
from diffusers.utils import load_image, export_to_video
from reproj_nus_scene_movecam import build_scene_clip,get_one_data,proj_movecam_frame1,save_cycle_proj
from glob import glob
import pickle 
import random
import imageio
import numpy as np
from PIL import Image
import torch.nn.functional as F
import math
import json
import tqdm
import random
import cv2
from copy import deepcopy
from PIL import Image, ImageDraw
import matplotlib

import torch.multiprocessing as mp
from functools import partial
import open3d as o3d
import argparse
from utils.image_util import depth2color
parser = argparse.ArgumentParser()

# running configurations
parser.add_argument('--model_path', type=str,default='./DiST_S/ckpt/')
parser.add_argument('--save_root', type=str,default = './DiST_S/reproj_nus/cycle_reproj/')
parser.add_argument('--video_length', type=int, default=6) # batch frame num
parser.add_argument('--multi_process', type=int, default=1) # mp num for inference
# parser.add_argument("--pesudo_flag_name",type=str,default=None,help="pesudo_flag_name")
# parser.add_argument("--front_only",action="store_true",default=False,help="whether to randomly flip images horizontally",)
parser.add_argument("--with_valid_mask",action="store_true",default=False,help="whether to use vlaid mask as input condition",)
parser.add_argument(
        "--video_height",
        type=int,
        default=320,
        help="Target height of resized video frames.",
    )
parser.add_argument(
        "--video_width",
        type=int,
        default=512,
        help="Target width of resized video frames.",
    )

parser.add_argument(
        "--is_trainset",
        action="store_true",
    )
parser.add_argument(
        "--mod_y_d",
        type=int,
        default=0,
    )

args = parser.parse_args()

def mask_pseudo_image(pesudo_image, random_aug_ratio=0.65, mask_aug_shape_ratio=0.):
        
        np_img = np.array(pesudo_image)
        ori_mask = (np.sum(np_img, axis=-1) == 0).astype(np.uint8)*255

        # debug_path = f"/data/gen_work/code/FreeVS/diffusers/work_dirs/debug/"
        # Image.fromarray(np_img).save(f"{debug_path}/debug_original.jpg")
        
        # random mask pixels
        new_mask_random = np.zeros_like(ori_mask)[:,:]
        rand = np.random.rand(*new_mask_random.shape)
        new_mask_random[rand < random_aug_ratio] = 1

        # mask shape
        predefined_shapes = [
            [(10, 10), (50, 10), (40, 50), (20, 40)],
            [(15, 15), (45, 15), (45, 45), (15, 45)],
            [(30, 5), (55, 30), (30, 55), (5, 30)],
            [(20, 10), (40, 10), (50, 30), (40, 50), (20, 50), (10, 30)],
            [(25, 5), (35, 5), (45, 25), (35, 45), (25, 45), (15, 25)],
            [(10, 20), (30, 10), (50, 20), (40, 40), (20, 40)],
            [(5, 30), (25, 10), (45, 30), (35, 50), (15, 50)],
            [(20, 20), (40, 20), (40, 40), (20, 40)],
        ]
        large_shapes = [
            [(25, 15), (35, 15), (45, 30), (35, 45), (25, 45), (15, 30)],
            [(10, 25), (30, 10), (50, 25), (40, 40), (20, 40)],
            [(10, 10), (100, 10), (90, 90), (20, 80)],
            [(20, 20), (90, 20), (90, 80), (20, 80)],
            [(30, 5), (80, 30), (30, 95), (5, 30)],
            [(15, 15), (95, 15), (95, 85), (15, 85)],
            [(25, 10), (85, 10), (95, 50), (85, 90), (25, 90), (10, 50)],
            [(10, 20), (50, 10), (100, 20), (80, 80), (20, 80)],
            [(5, 30), (50, 10), (105, 30), (85, 70), (15, 70)],
            [(20, 20), (90, 20), (90, 80), (20, 80)],
            [(25, 15), (85, 15), (95, 50), (85, 85), (25, 85), (15, 50)],
            [(10, 25), (50, 10), (100, 25), (80, 75), (20, 75)]]
        
        mask_shape = np.zeros_like(ori_mask)[:,:]
        random_var = np.random.rand()
        if random_var < mask_aug_shape_ratio:
            
            mask_image = Image.fromarray((mask_shape * 255).astype(np.uint8))
            draw = ImageDraw.Draw(mask_image)
            width, height = mask_image.size

            num_shapes_to_add = random.randint(1, 5)
            selected_shapes = random.sample(predefined_shapes, num_shapes_to_add)
            num_large_shapes_to_add = random.randint(1, 3)
            selected_shapes += random.sample(large_shapes, num_large_shapes_to_add)

            for shape in selected_shapes:
                max_x = width - 110
                max_y = height - 110
                top_left_x = random.randint(0, max_x)
                top_left_y = random.randint(0, max_y)    
                translated_shape = [(x + top_left_x, y + top_left_y) for (x, y) in shape]    
                draw.polygon(translated_shape, fill=1)

            mask_shape = np.array(mask_image) #// 255

        np_img[new_mask_random == 1, :] = 0
        np_img[mask_shape == 1, :] = 0
        
        # if not os.path.exists(debug_path):
        #     os.makedirs(debug_path)
        # Image.fromarray(np_img).save(f"{debug_path}/debug_mask_aug.jpg")
        # exit()

        return Image.fromarray(np_img)

def pros_depth_(depth,depth_min=0.5, depth_max=100.0):
        depth = np.clip(depth, depth_min, depth_max)
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        depth /= depth_max

        return depth

def load_depth(file_name, depth_min=0.5, depth_max=100.0):
        if "gt" in file_name:
            depth = np.load(file_name)['depth_gt']
        else:
            depth = np.load(file_name)['depth_proj']
        depth = np.clip(depth, depth_min, depth_max)
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        depth /= depth_max

        return depth

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def process_scene(sample_idxs, scene_idx , save_root, data_infos,sample_token2depth,mod_y_d,gpu_id, args):
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    model_path = args.model_path
    # img_pickle = args.img_pickle
    sample_infos =[]
    scene_len= len(sample_idxs)
    interval=2
    sample_pesudo_list = []
    for idx in range(0,scene_len,interval):
        sample_idx = sample_idxs[idx]
        sample_info = get_one_data(data_infos,sample_token2depth,sample_idx)
        sample_infos.append(sample_info)
        sample_pesudo_list.append(sample_info["token"])

    
    # img_root = args.img_root
    # pseudo_root = os.path.join(img_root, args.pesudo_flag_name)
    # GT_root = os.path.join(img_root, 'frame_gt')
    # mod_y_d = args.mod_y_d
    # interval=2
    scene_use_len = len(sample_infos)
    

        # sample_pesudo_list.append(sample_infos[i]["token"])

    pipe = StableVideoDiffusionPipeline_convnb_multiframe_mm.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16", device_map={"": f"cuda:{gpu_id}"}
    )
    pipe.to(f"cuda:{gpu_id}")
    pipe.enable_model_cpu_offload(gpu_id)

    # scenes = get_val_scene_list()

    # scenes = ['12505030131868863688_1740_000_1760_000_FRONT'] #demo case

    video_length=args.video_length

    # savepath=args.output_dir
    # savepath=f"{args.output_dir}/{args.pesudo_flag_name}"
    # os.makedirs(savepath, exist_ok=True)

    
    cnt = 0
    # camera_view = ['CAM_FRONT', 'CAM_BACK','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK_LEFT']
    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
    # save_pesudo_root = f"/data/code/FreeVS/vis_cycle_proj/1/y+{mod_y_d}_reproj"
    save_pesudo_root = f"{save_root}/{scene_idx}"

    for i in tqdm.tqdm(range(0,math.floor(scene_use_len/video_length)),desc=f"scene{scene_idx}-mody{mod_y_d}"):
    # for i in range(0,1):
        scene_rgb_pesudo_list = [] # t cam
        scene_depth_pesudo_list = []
        for sample_idx in range(video_length*i,video_length*i+video_length):
            target_proj_rgb_list,target_proj_depth_list = proj_movecam_frame1(deepcopy(sample_infos[sample_idx]),mod_y_d=mod_y_d)  #6cam
            # prisnt(target_proj_rgb_list[0].shape)
            scene_rgb_pesudo_list.append(target_proj_rgb_list) #new view conditoin
            scene_depth_pesudo_list.append(target_proj_depth_list) #new view depth
        # cnt +=1
        # if cnt%6==0:
        #     cnt = 0

        rgb_frames_cam_t = [] # cam t
        depth_frames_cam_t = []
        for cam_idx in range(6): #6cam
            pseudo_images = []
            pseudo_depths = []
            for t in range(video_length):
                pseudo_image = scene_rgb_pesudo_list[t][cam_idx]
                # pseudo_image = load_image(pseudo_image)
                pseudo_image = pseudo_image.resize((int(args.video_width),int(args.video_height)))
                pseudo_images.append(pseudo_image)

                pseudo_depth = scene_depth_pesudo_list[t][cam_idx]
                pseudo_depth = pros_depth_(pseudo_depth)
                pseudo_depth = cv2.resize(pseudo_depth, (args.video_width,args.video_height), interpolation=cv2.INTER_NEAREST)
                if args.with_valid_mask:
                    valid_mask = (pseudo_depth[:,:,-1] > 0.5/100.).astype(np.float32)
                    pseudo_depth[:,:,-1] = valid_mask
                    
                pseudo_depth = np.transpose(pseudo_depth, (2,0,1))

                pseudo_depths.append(pseudo_depth)
                # print(pseudo_depth.shape)

            # import pdb; pdb.set_trace()

            origin_image = deepcopy(pseudo_images[0])
            # origin_image = gtimage_names[0]
            # origin_image = load_image(origin_image)
            # origin_image = origin_image.resize((int(args.video_width),int(args.video_height)))

            # origin_image = origin_image.resize((int(960),int(384)))

            # all_gt += [origin_image]
            # all_pseudo += [image]

            generator = torch.manual_seed(42)

            ret_all = pipe(pseudo_images, pseudo_depths,origin_image=origin_image,
                            width=origin_image.size[0], height=origin_image.size[1],
                            num_frames=video_length, num_inference_steps=25,
                            min_guidance_scale=2.0,max_guidance_scale=2.0,
                            noise_aug_strength=0.02, generator=generator)
            
            frames = ret_all.frames[0] # generate rgb
            depth_frames = ret_all.depth_frames[0] #generate depth

            rgb_frames_cam_t.append(frames)
            depth_frames_cam_t.append(depth_frames)
        # scene_rgb_pesudo_list = []
        # scene_depth_pesudo_list = []

        for t in range(video_length):
            
            datainfo = sample_infos[video_length*i+t]
            sample_token = datainfo["token"]

            ''' condition vis
            for cam_idx in range(6):
                save_cam_reproj_path = f"{save_pesudo_root}/{vis_cameras[cam_idx]}/reproj_vis/"
                os.makedirs(save_cam_reproj_path,exist_ok=True)
                scene_rgb_pesudo_list[t][cam_idx].save(f"{save_cam_reproj_path}/{sample_token}_rgb_C.jpg")
                dep_np = scene_depth_pesudo_list[t][cam_idx]
                dep_vis = depth2color(dep_np)
                dep_vis.save(f"{save_cam_reproj_path}/{sample_token}_depth_C.jpg")

                rgb_novel_gen = rgb_frames_cam_t[cam_idx][t]
                depth_novel_gen = depth_frames_cam_t[cam_idx][t]
                rgb_novel_gen.save(f"{save_cam_reproj_path}/{sample_token}_rgb_Gen.jpg")
                depth_novel_vis = depth2color(np.mean(depth_novel_gen, axis=-1) * 100,100) 
                depth_novel_vis.save(f"{save_cam_reproj_path}/{sample_token}_depth_Gen.jpg")
            '''



            
            # for cam_idx in range(6): #6cam

            rgb_t_list = [sublist[t] for sublist in rgb_frames_cam_t]   
            depth_t_list = [sublist[t] for sublist in depth_frames_cam_t]
            save_cycle_proj(datainfo,(int(args.video_width),int(args.video_height)),rgb_t_list,depth_t_list,save_pesudo_root,mod_y_d=mod_y_d)
            # reproj_cyc_rgb, reproj_cyc_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, new_width, new_height)

    with open(f"{save_pesudo_root}/pesudo_sampletoken.json",'w') as file:
        json.dump(sample_pesudo_list,file)



    
                

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    with open("../../misc/nus_sampletoken2depthroot.json",'r') as file:
        sample_token2depth = json.load(file) 

    nus_root = "/data/nuscenes"
    train_pkl_root = f"{nus_root}/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"
    val_pkl_root = f"{nus_root}/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"
    if args.is_trainset:
        pkl_root = train_pkl_root
        split_set="train"
    else:
        pkl_root = val_pkl_root
        split_set="val"

    with open(pkl_root, 'rb') as file:
        # 使用pickle.load()从文件中读取序列化的对象
        anno_data = pickle.load(file)

    data_infos = list(sorted(anno_data["infos"], key=lambda e: e["timestamp"]))
    clip_infos = build_scene_clip(data_infos, anno_data['scene_tokens'])

    model_path = args.model_path
    scene_idxs = range(0,700)
    
    # sample_idxs = clip_infos[scene_idx]

    save_cycle_proj_root = f"{args.save_root}/{split_set}"
    # img_pickle = args.img_pickle
    # img_root = args.img_root
    # pseudo_root = os.path.join(img_root, args.pesudo_flag_name)
    # GT_root = os.path.join(img_root, 'frame_gt')


    camera_view = ['CAM_FRONT', 'CAM_BACK','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK_LEFT']
    # camera_view = ['CAM_FRONT']
    num_gpus = torch.cuda.device_count()
    processes_per_gpu = args.multi_process  # 每个GPU上运行的进程数
    
    # all_tasks = [(scene, cam) for scene in scenes for cam in camera_view]
    all_tasks = [(scene_idx,clip_infos[scene_idx]) for scene_idx in scene_idxs]
    processes = []
    # mod_y_d = 2
    # process_scene(sample_idxs,scene_idx,save_cycle_proj_root, data_infos,sample_token2depth,mod_y_d,0, args)

    myd_list = [-3, -2, -1, 1, 2,3]  # 示例列表
    # myd_list = [-3]
    print(f" ====! Total {len(all_tasks)} tasks !====")
    for i, (scene_idx,sample_idxs) in enumerate(all_tasks):
        # print(sample_idxs)
        gpu_id = i % num_gpus
        mod_y_d = random.choice(myd_list)
        p = mp.Process(
            target=process_scene,
            args=(sample_idxs, scene_idx,save_cycle_proj_root, data_infos,sample_token2depth,mod_y_d,gpu_id, args)
        )
        processes.append(p)
        p.start()
        
        if len(processes) >= num_gpus * processes_per_gpu:
            for p in processes:
                p.join()
            processes = []
    
    for p in processes:
        p.join()
