import sys
import torch
import os
# from src.pipelines.pipeline_stable_video_diffusion_custom import StableVideoDiffusionPipeline_convnb_multiframe
from src.pipelines.pipeline_stable_video_diffusion_custom_mm import StableVideoDiffusionPipeline_convnb_multiframe_mm
from diffusers.utils import load_image, export_to_video
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
from PIL import Image, ImageDraw
import matplotlib

import torch.multiprocessing as mp
from functools import partial

import argparse
parser = argparse.ArgumentParser()

# running configurations
parser.add_argument('--model_path', type=str,default='./DiST_S/ckpt/')
parser.add_argument('--img_root', type=str,default = './DiST_S/reproj_nus/val')
parser.add_argument('--output_dir', type=str, default='work_dirs/DiST_output') 
parser.add_argument('--video_length', type=int, default=6) # batch frame num
parser.add_argument('--multi_process', type=int, default=2) # mp num for inference
parser.add_argument("--pesudo_flag_name",type=str,default=None,help="pesudo_flag_name")
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


def process_camera(scene_, cam_name, gpu_id, args):
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    model_path = args.model_path
    # img_pickle = args.img_pickle
    img_root = args.img_root
    pseudo_root = os.path.join(img_root, args.pesudo_flag_name)
    GT_root = os.path.join(img_root, 'frame_gt')
    
    pipe = StableVideoDiffusionPipeline_convnb_multiframe_mm.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16", device_map={"": f"cuda:{gpu_id}"}
    )
    pipe.to(f"cuda:{gpu_id}")
    pipe.enable_model_cpu_offload(gpu_id)

    # scenes = get_val_scene_list()

    # scenes = ['12505030131868863688_1740_000_1760_000_FRONT'] #demo case

    video_length=args.video_length

    # savepath=args.output_dir
    savepath=f"{args.output_dir}/{args.pesudo_flag_name}"
    os.makedirs(savepath, exist_ok=True)

    for scene in [scene_]:
        token_josn = os.path.join(pseudo_root, scene, 'pesudo_sampletoken.json')
        with open(token_josn, 'r') as f:
            sample_token_list = json.load(f)
        
        scene_sample_len = len(sample_token_list)

        allframes, alldepth_frames, allcombined = [], [], []
        count=0

        os.makedirs(os.path.join(savepath, scene, cam_name),exist_ok=True)

        for rand_int in tqdm.tqdm(range(0, math.floor(scene_sample_len/video_length)), 
                                    desc=f'GPU-{gpu_id} Infer {cam_name} of {scene}'): #5hz

            if rand_int < math.floor(scene_sample_len/video_length):
                images = [os.path.join(pseudo_root, scene, cam_name, f"{sample_token_list[i]}.jpg") 
                            for i in range(rand_int*video_length, rand_int*video_length+video_length)]
                gtimage_names = [os.path.join(GT_root, scene, cam_name, f"{sample_token_list[i]}.jpg") 
                                    for i in range(rand_int*video_length, rand_int*video_length+video_length)]
                depths = [os.path.join(pseudo_root, scene, cam_name, f"{sample_token_list[i]}.npz") 
                            for i in range(rand_int*video_length, rand_int*video_length+video_length)]
                gtdepth_names = [os.path.join(GT_root, scene, cam_name, f"{sample_token_list[i]}.npz") 
                                    for i in range(rand_int*video_length, rand_int*video_length+video_length)]
            else:
                images = [os.path.join(pseudo_root, scene, cam_name, f"{sample_token_list[i]}.jpg") 
                            for i in range(scene_sample_len-video_length, scene_sample_len)]
                gtimage_names = [os.path.join(GT_root, scene, cam_name, f"{sample_token_list[i]}.jpg") 
                                    for i in range(scene_sample_len-video_length, scene_sample_len)]
                depths = [os.path.join(pseudo_root, scene, cam_name, f"{sample_token_list[i]}.npz") 
                            for i in range(scene_sample_len-video_length, scene_sample_len)]
                gtdepth_names = [os.path.join(GT_root, scene, cam_name, f"{sample_token_list[i]}.npz") 
                                    for i in range(scene_sample_len-video_length, scene_sample_len)]
            
            if not sum([os.path.exists(image) for image in images]) == len(images):
                break

            pseudo_images, gt_images, pseudo_depths, gt_depths = [], [], [], []
            for image, gt_image, depth, gt_depth in zip(images, gtimage_names, depths, gtdepth_names):
                image = load_image(image)
                gt_image = load_image(gt_image)

                depth = load_depth(depth, depth_min=0.5, depth_max=100.0)
                gt_depth = load_depth(gt_depth, depth_min=0.5, depth_max=100.0)

                ## NOTE: mask the pseudo image
                # image = mask_pseudo_image(image, random_aug_ratio=0.25, mask_aug_shape_ratio=0.)

                # image = image.resize((int(960),int(384)))
                image = image.resize((int(args.video_width),int(args.video_height)))
                gt_image = gt_image.resize((int(args.video_width),int(args.video_height)))

                depth = cv2.resize(depth, (args.video_width,args.video_height), interpolation=cv2.INTER_NEAREST)
                gt_depth = cv2.resize(gt_depth, (args.video_width,args.video_height), interpolation=cv2.INTER_NEAREST)

                if args.with_valid_mask:
                    valid_mask = (depth[:,:,-1] > 0.5/100.).astype(np.float32)
                    depth[:,:,-1] = valid_mask
                    
                depth = np.transpose(depth, (2,0,1))
                gt_depth = np.transpose(gt_depth, (2,0,1))

                pseudo_images.append(image)
                gt_images.append(gt_image)
                pseudo_depths.append(depth)
                gt_depths.append(gt_depth)
        

            # origin_image = imgs[rand_int*(video_length)]['image']
            origin_image = gtimage_names[0]
            origin_image = load_image(origin_image)
            origin_image = origin_image.resize((int(args.video_width),int(args.video_height)))
            # origin_image = origin_image.resize((int(960),int(384)))

            # all_gt += [origin_image]
            # all_pseudo += [image]

            generator = torch.manual_seed(42)

            ret_all = pipe(pseudo_images, pseudo_depths,origin_image=origin_image,
                            width=origin_image.size[0], height=origin_image.size[1],
                            num_frames=video_length, num_inference_steps=25,
                            min_guidance_scale=2.0,max_guidance_scale=2.0,
                            noise_aug_strength=0.02, generator=generator)
            
            frames = ret_all.frames[0]
            depth_frames = ret_all.depth_frames[0]  # [6,467,768,3]

            # unscale depth prediction
            decode_depth_frames, colored_depth_frames, colored_gt_depths, colored_pseudo_depths = [], [], [], []
            for i in range(depth_frames.shape[0]):
                dep = np.mean(depth_frames[i], axis=-1) * 100.
                decode_depth_frames.append(dep)
                dep_color = colorize_depth_maps(dep, min_depth=0.5, max_depth=100.)[0]
                colored_depth_frames.append(np.transpose(dep_color*255, (1,2,0)).astype(np.uint8))

                dep_pseudo = np.mean(pseudo_depths[i][:2,:,:], axis=0) * 100.
                dep_pseudo_color = colorize_depth_maps(dep_pseudo, min_depth=0.5, max_depth=100.)[0]
                colored_pseudo_depths.append(np.transpose(dep_pseudo_color*255, (1,2,0)).astype(np.uint8))

                dep_gt = np.mean(gt_depths[i], axis=0) * 100.
                dep_gt_color = colorize_depth_maps(dep_gt, min_depth=0.5, max_depth=100.)[0]
                colored_gt_depths.append(np.transpose(dep_gt_color*255, (1,2,0)).astype(np.uint8))


            # import pdb; pdb.set_trace()
            allframes = allframes + frames
            alldepth_frames = alldepth_frames + decode_depth_frames

            for _i,frame in enumerate(frames):
                export_path = os.path.join(savepath, scene, cam_name, os.path.split(gtimage_names[_i])[-1])
                combined_image = np.zeros((frame.size[1] * 2, frame.size[0] * 3, 3), dtype=np.uint8)
                
                frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) 
                pseudo_cv = cv2.cvtColor(np.array(pseudo_images[_i]), cv2.COLOR_RGB2BGR) 
                gt_cv = cv2.cvtColor(np.array(gt_images[_i]), cv2.COLOR_RGB2BGR) 
                
                # RGB
                combined_image[0:frame.size[1], 0:frame.size[0]] = frame_cv
                combined_image[0:frame.size[1], frame.size[0]:frame.size[0]*2] = pseudo_cv
                combined_image[0:frame.size[1], frame.size[0]*2:frame.size[0]*3] = gt_cv
                
                # Depth
                combined_image[frame.size[1]:frame.size[1]*2, 0:frame.size[0]] = cv2.cvtColor(colored_depth_frames[_i], cv2.COLOR_RGB2BGR) 
                combined_image[frame.size[1]:frame.size[1]*2, frame.size[0]:frame.size[0]*2] = cv2.cvtColor(colored_pseudo_depths[_i], cv2.COLOR_RGB2BGR)
                combined_image[frame.size[1]:frame.size[1]*2, frame.size[0]*2:frame.size[0]*3] = cv2.cvtColor(colored_gt_depths[_i], cv2.COLOR_RGB2BGR)
                
                allcombined += [combined_image]
                frame.save(export_path) # for fid eval
                # cv2.imwrite(export_path, combined_image)
                count += 1

        filename = os.path.join(savepath, scene, cam_name)+'.mp4'
        fps = 10
        with imageio.get_writer(filename,fps=fps,codec='libx264', bitrate='5000k', quality=10) as video:
            # for frame in allframes:
            for frame in allcombined:
                # frame = frame.convert('RGB')
                # frame = Image.fromarray(frame)#.convert('RGB')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video.append_data(np.array(frame))
                

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    model_path = args.model_path
    # img_pickle = args.img_pickle
    img_root = args.img_root
    pseudo_root = os.path.join(img_root, args.pesudo_flag_name)
    GT_root = os.path.join(img_root, 'frame_gt')

    # scenes = sorted(os.listdir(pseudo_root)[300:500])
    # scenes = [str(i) for i in range(500,600)]
    # scenes = [str(num) for num in range(0,30,1)]
    
    scenes = range(0,150)
    camera_view = ['CAM_FRONT', 'CAM_BACK','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK_LEFT']
    # camera_view = ['CAM_FRONT']
    num_gpus = torch.cuda.device_count()
    processes_per_gpu = args.multi_process 
    
    all_tasks = [(scene, cam) for scene in scenes for cam in camera_view]
    processes = []
    
    print(f" ====! Total {len(all_tasks)} tasks !====")
    for i, (scene, cam_name) in enumerate(all_tasks):
        gpu_id = i % num_gpus
        
        p = mp.Process(
            target=process_camera,
            args=(scene, cam_name, gpu_id, args)
        )
        processes.append(p)
        p.start()
        
        if len(processes) >= num_gpus * processes_per_gpu:
            for p in processes:
                p.join()
            processes = []
    
    for p in processes:
        p.join()
