import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
# from tqdm.auto import tqdm
from tqdm import tqdm
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
import cv2
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.models.projection import My_proj
from transformers import CLIPVisionModelWithProjection
from inference.depthlab_pipeline import DepthLabPipeline
from utils.seed_all import seed_all
from utils.image_util import get_filled_for_latents, save_depth_color_video,colorize_depth_maps, chw2hwc,resize_max_res_cv2
import json
import matplotlib.pyplot as plt

def load_and_process_mask(mask_path):
    image = Image.open(mask_path).convert('L')
    mask = np.array(image)
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return mask

def generate_occu_mask(depth_map):
    occu_mask = np.zeros_like(depth_map)
    
    points = np.argwhere(depth_map != 0)
    
    for point in points:
        i, j = point
        radius = int(1000 / depth_map[i,j]**2)
        # 使用OpenCV的circle函数在mask上绘制圆形区域
        cv2.circle(occu_mask, (j, i), radius, 1, -1)
    
    return occu_mask

def depth2color(depth_numpy,clip_max=100):
    depth_numpy = depth_numpy.clip(min=0.,max=clip_max)

    # for nus 0-100m ?
    depth_pred_colored = colorize_depth_maps(
        depth_numpy, 0, clip_max, cmap="Spectral").squeeze() # [3, H, W], value in (0, 100)
    
    depth_pred_colored = (depth_pred_colored * 255).astype(np.uint8)
    depth_pred_colored = Image.fromarray(chw2hwc(depth_pred_colored))
    return depth_pred_colored

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--normalize_scale",
        type=float,
        default=1,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--denoising_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--mapping_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--reference_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--scene_root", type=str, required=True, help="Path to nus scene."
    )

    parser.add_argument(
        "--blend",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--use_mvs_pcd",
        action="store_true",
        help="Whether or not to use MVS pcd",
    )

    parser.add_argument('--cam_use', nargs='+', type=int)

    args = parser.parse_args()
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    processing_res = args.processing_res
    seed = args.seed
    scene_root = args.scene_root
    
    
    vis_cameras_full = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
    vis_cameras = [vis_cameras_full[i] for i in args.cam_use]
    # vis_cameras = ['CAM_BACK']
    # vis_cameras = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # vis_cam = "CAM_FRONT"

    # output_dir_color = os.path.join(output_dir, "depth_colored")
    # output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_dir_color, exist_ok=True)
    # os.makedirs(output_dir_npy, exist_ok=True)

    logging.info(f"output dir = {output_dir}")

    # if args.input_image_paths is not None:
    #     assert len(args.input_image_paths) == len(args.known_depth_paths)
    #     assert len(args.input_image_paths) == len(args.masks_paths)
    # input_image_paths = args.input_image_paths
    # known_depth_paths = args.known_depth_paths
    # masks_paths = args.masks_paths
    

    print(f"arguments: {args}")
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
    denoising_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=12, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    reference_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=4, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    mapping_layer=My_proj()


    mapping_layer.load_state_dict(
        torch.load(args.mapping_path, map_location="cpu"),
        strict=False,
        )
    mapping_device = torch.device("cuda")
    mapping_layer.to(mapping_device )
    reference_unet.load_state_dict(
                torch.load(args.reference_unet_path, map_location="cpu"),
        )
    denoising_unet.load_state_dict(
        torch.load(args.denoising_unet_path, map_location="cpu"),
        strict=False,
        )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    pipe = DepthLabPipeline(reference_unet=reference_unet,
                                       denoising_unet = denoising_unet,  
                                       mapping_layer=mapping_layer,
                                       vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       image_enc=image_enc,
                                       scheduler=scheduler,
                                       ).to('cuda')
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    # -------------------- Inference and saving --------------------
    use_mvs_pcd = args.use_mvs_pcd
    with open(os.path.join(scene_root,"sample_token_list.json"),'r') as file:
        sample_token_list = json.load(file)

    max_scene_len = len(sample_token_list)
    vis_time = list(range(0, max_scene_len, 1))#[0,5,15,20,25,30,35,40,45,50]

    # vis_time = list(range(135, max_scene_len, 1))
    # vis_time = [0,5,10,15,20,25,30,35]
    with torch.no_grad():
        cam6_r_d_color_list = []

        for vis_cam in vis_cameras:
            depth_color_img_list = []
            depth_color_img_list2 = []
            rgb_depth_color_list = []
            for vis_i in tqdm(vis_time,desc=f"Processing {vis_cam}"):
                sample_token = sample_token_list[vis_i]
                print(sample_token)
                rgb_depth_name_base = f"rgb_depth_{vis_i}"

                input_npz_path = os.path.join(scene_root,sample_token,vis_cam,f"rgb_depth_{vis_i}.npz")
                # save path
                rgb_depth_data = np.load(input_npz_path)

                input_image = Image.fromarray(rgb_depth_data['rgb'])
                depth_numpy = rgb_depth_data['raw_depth']
                if 'mvs_depth' in rgb_depth_data.files:
                    mvs_depth_numpy = rgb_depth_data['mvs_depth']
                else:
                    use_mvs_pcd = False

                # pred_name_base = rgb_name_base + "_pred"
                # input_npz_path = os.path.join(scene_root,sample_token,vis_cam,f"rgb_depth_{vis_i}.npz")
                npy_save_path = os.path.join(output_dir,sample_token,vis_cam,f"refined_depth.npz")

                # colored_save_path = os.path.join(
                #     output_dir_color, f"{pred_name_base}_colored.png"
                # )
                # input_image = Image.open(input_image_path)
                # depth_numpy=np.load(known_depth_path)

                lidar_depth_color = depth2color(depth_numpy.copy())
                
                mask=np.zeros(depth_numpy.shape)
                mask[depth_numpy>0.05]=0
                mask[depth_numpy<0.05]=1
                
                # import pdb; pdb.set_trace()
                
                

                if args.refine is not True and use_mvs_pcd==False:
                    depth_numpy=get_filled_for_latents(mask,depth_numpy)
                
                mask_inter = mask.copy()
                if use_mvs_pcd:
                    # mvs_depth_path = f"{scene_root}/depth_{vis_i}_{vis_cam}_mvs.npy"
                    # mvs_depth_numpy=np.load(mvs_depth_path)
                    # print(mvs_depth_numpy[30:50,30:50])
                    mask_mvs=np.zeros(mvs_depth_numpy.shape)
                    mask_mvs[mvs_depth_numpy>0.05]=0 #ko
                    mask_mvs[mvs_depth_numpy<0.05]=1

                    mask_occu=generate_occu_mask(depth_numpy)
                    mask_fill_mvs=(mask==1) & (mask_mvs==0) & (mask_occu==0)
                    depth_numpy[mask_fill_mvs] = mvs_depth_numpy[mask_fill_mvs]

                    
                    mask_inter[mask_fill_mvs==1]=0
                    depth_numpy=get_filled_for_latents(mask_inter,depth_numpy)
                


                '''
                Semantic_path = os.path.join(scene_root,sample_token,vis_cam,f"semantic.npz")
                Semantic = np.load(Semantic_path)['sem'] #225*400
                Semantic_mask = resize_max_res_cv2(Semantic , max_edge_resolution=processing_res, interpolation=cv2.INTER_NEAREST )
                sky_idx = 10
                sky_mask = np.isin(Semantic_mask, sky_idx) 
                Rep_mask = sky_mask & mask_fill_mvs
                depth_numpy[Rep_mask] = 100
                '''
                
                gds = 1
                # warp_range = 30

                '''
                if vis_i != 0:
                    mask1 = mask.copy()
                    e_kernel = np.ones((3, 3), np.uint8)
                    eroded_mask = cv2.erode(mask1, e_kernel, iterations=3)
                    mask1 = eroded_mask
                    # plt.clf()
                    # plt.imshow(mask1)
                    # plt.savefig("eroded_mask.png")
                    mask2 = last_depth_pred > warp_range
                    mask3 = np.zeros(depth_numpy.shape,dtype=bool)
                    height = depth_numpy.shape[0]
                    mask3[:height//2, :] = 1

                    prop_mask = np.logical_and(np.logical_or(mask2,mask3),mask1)

                    plt.clf()
                    plt.imshow(prop_mask)
                    plt.savefig("prop_mask.png")

                    depth_numpy[prop_mask] = last_depth_pred[prop_mask]
                '''
                if np.sum(mask==0)==0: #no lidar point
                    mask = mask_mvs
                    print("no lidar points! use mvs points!")
                pipe_out = pipe(
                    input_image,
                    denosing_steps = denoise_steps,
                    processing_res = processing_res,
                    match_input_res = True,
                    batch_size =1,
                    color_map = "Spectral",
                    show_progress_bar = False,
                    depth_numpy_origin = depth_numpy,
                    mask_origin = mask,
                    guidance_scale = gds,
                    normalize_scale = args.normalize_scale,

                    strength = args.strength,
                    blend = args.blend)

                depth_pred: np.ndarray = pipe_out.depth_np
                depth_color_img: Image.Image = pipe_out.depth_colored
                # depth_color_img_list.append(depth_color_img)

                last_depth_pred = depth_pred

                # if os.path.exists(colored_save_path):
                #     logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")
                #np.save(npy_save_path,depth_pred)
                # pipe_out.depth_colored.save(colored_save_path)

                # mask_2 = mask.copy()
                # mask_2[mask_occu==1]==0
                
                pipe_out2 = pipe(
                    input_image,
                    denosing_steps = denoise_steps,
                    processing_res = processing_res,
                    match_input_res = True,
                    batch_size =1,
                    color_map = "Spectral",
                    show_progress_bar = False,
                    depth_numpy_origin = depth_pred,
                    mask_origin = mask_inter,
                    guidance_scale = gds,
                    normalize_scale = args.normalize_scale,

                    strength = args.strength,
                    blend = args.blend)

                depth_pred2: np.ndarray = pipe_out2.depth_np
                depth_color_img2: Image.Image = pipe_out2.depth_colored

               

                np.savez_compressed(npy_save_path,depth_pred=depth_pred2) 
                vis_depth_path = os.path.join(output_dir,sample_token,vis_cam,f"vis_rgb_Rdepth.jpg")

                resz_input_image = input_image.resize((depth_color_img2.width, depth_color_img2.height))
                vis_w,vis_h = depth_color_img2.width, depth_color_img2.height
                # new_image = Image.new('RGB', (vis_w, vis_h*2))
                # new_image.paste(resz_input_image, (0, 0))  
                # new_image.paste(depth_color_img2, (0, vis_h)) 
                # new_image.save(vis_depth_path)
                # pipe_out2.depth_colored.save(colored_save_path2)
                
                # vis ori
                # depth_pros_color = depth2color(depth_numpy)
                resz_input_image = input_image.resize((depth_color_img2.width, depth_color_img2.height))

                vis_w,vis_h = depth_color_img2.width, depth_color_img2.height

                if use_mvs_pcd:
                    mvs_depth_color = depth2color(mvs_depth_numpy)
                    h_len = 4
                else:
                    h_len = 3

                new_image = Image.new('RGB', (vis_w, vis_h*h_len))
                new_image.paste(resz_input_image, (0, 0))  
                new_image.paste(depth_color_img2, (0, vis_h))  
                new_image.paste(lidar_depth_color,(0, vis_h*2))
                if use_mvs_pcd:
                    new_image.paste(mvs_depth_color, (0, vis_h*3)) 
                new_image.save(vis_depth_path)
                # rgb_depth_color_list.append(new_image)
                

            # cam6_r_d_color_list.append(rgb_depth_color_list)


            # save_depth_color_video(depth_color_img_list,os.path.join(output_dir,f"depth_color_video_{vis_cam}_{warp_range}.mp4"),fps=6)
            # save_depth_color_video(depth_color_img_list2,os.path.join(output_dir,f"refine_depth_color_video_{vis_cam}_{warp_range}.mp4"),fps=6)
            # save_depth_color_video(rgb_depth_color_list,os.path.join(output_dir,f"refine_rgb+depth_{vis_cam}_{warp_range}.mp4"),fps=6)
            # save_depth_color_video(rgb_depth_color_list,os.path.join(output_dir,f"refine_5img_{vis_cam}.mp4"),fps=6)
        



            