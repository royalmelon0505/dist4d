import os
import json
import numpy as np
import torch
from PIL import Image,ImageEnhance, ImageStat
# from tqdm.auto import tqdm
from tqdm import tqdm
# from utils.image_util import resize_max_res, resize_max_res_cv2,save_depth_color_video,get_filled_for_latents
import cv2
# from infer_nus_video_mp import depth2color
from utils.image_util import colorize_depth_maps,depth2color
from scipy.ndimage import distance_transform_edt

import pickle
from pyquaternion import Quaternion
import open3d as o3d
from copy import deepcopy
from tqdm import tqdm
import argparse
import sys



Rdepth_root = "./dist4d/depth_process/nus_Rdepth/"



def build_scene_clip(data_infos,scene_tokens):
    token_data_dict = {
        item['token']: idx for idx, item in enumerate(data_infos)}
    all_clips = []
    for sid, scene in enumerate(scene_tokens):
        clip = [token_data_dict[token] for token in scene]
        all_clips.append(clip)
    return all_clips

def get_one_data(data_infos,sample_token2depth,index):
    info = data_infos[index]
    sample_token = info["token"]
    sample_Rdepth_root = f"{Rdepth_root}{sample_token2depth[sample_token]}"

    data = dict(
                token=info["token"],
                sample_idx=info['token'],
                lidar_path=info["lidar_path"],
                sweeps=info["sweeps"],
                timestamp=info["timestamp"],
                location=info["location"],
            )

    # ego to global transform
    ego2global = np.eye(4).astype(np.float32)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = info["ego2global_translation"]
    data["ego2global"] = ego2global

    # lidar to ego transform
    lidar2ego = np.eye(4).astype(np.float32)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = info["lidar2ego_translation"]
    data["lidar2ego"] = lidar2ego


    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
    # new_width, new_height = 400,224
    data["image_paths"] = []
    data["lidar2camera"] = []
    data["lidar2image"] = []
    data["camera2ego"] = []
    data["camera_intrinsics"] = []
    data["camera2lidar"] = []
    data["camera2global"] = []
    data["refine_depth_path"] = []
    data["semantic_path"] = []
    for cam in vis_cameras:
        camera_info = info['cams'][cam]
        data["image_paths"].append(camera_info["data_path"])

        # lidar to camera transform
        lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
        lidar2camera_t = (
            camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
        )
        lidar2camera_rt = np.eye(4).astype(np.float32)
        lidar2camera_rt[:3, :3] = lidar2camera_r.T
        lidar2camera_rt[3, :3] = -lidar2camera_t
        data["lidar2camera"].append(lidar2camera_rt.T)

        # camera intrinsics
        camera_intrinsics = np.eye(4).astype(np.float32)
        camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
        data["camera_intrinsics"].append(camera_intrinsics)

        # lidar to image transform
        lidar2image = camera_intrinsics @ lidar2camera_rt.T
        data["lidar2image"].append(lidar2image)

        # camera to ego transform
        camera2ego = np.eye(4).astype(np.float32)
        camera2ego[:3, :3] = Quaternion(
            camera_info["sensor2ego_rotation"]
        ).rotation_matrix
        camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
        data["camera2ego"].append(camera2ego)

        # camera to lidar transform
        camera2lidar = np.eye(4).astype(np.float32)
        camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
        data["camera2lidar"].append(camera2lidar)
        
        camera_ego2global = np.eye(4).astype(np.float32)
        camera_ego2global[:3, :3] = Quaternion(
            camera_info["ego2global_rotation"]
        ).rotation_matrix
        camera_ego2global[:3, 3] = camera_info["ego2global_translation"]
        camera2global = camera_ego2global @ camera2ego
        data["camera2global"].append(camera2global)

        sample_Rdepth_path = os.path.join(sample_Rdepth_root,cam,"refined_depth.npz")
        # sample_Rdepth = np.load(sample_Rdepth_path)['depth_pred'].astype(np.float32)
        # sample_Rdepth = cv2.resize(sample_Rdepth, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        data["refine_depth_path"].append(sample_Rdepth_path)
        Semantic_path = os.path.join(sample_Rdepth_root,cam,"semantic_oneformer.npz")
        data["semantic_path"].append(Semantic_path)
        # Semantic = np.load(Semantic_path)['sem'] #225*400
        # Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # sample_Rdepth[Semantic==27]=100 # sky mask for semantic

    # import pdb; pdb.set_trace()
    # print(ego)
    return data

def save_cycle_proj(datainfo,img_wh,rgb_t_list,depth_t_list,save_pesudo_root,mod_y_d=2):

    # gen_width, gen_height = img_wh[0],img_wh[1]
    pcd_width, pcd_height = 1200, 675
    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
    # CAM_BACK_MASK = Image.open("/data/code/DepthLab/CAM_BACK_mask.png")
    # CAM_BACK_MASK_np = np.asarray(CAM_BACK_MASK)
    # CAM_BACK_MASK_np = cv2.resize(CAM_BACK_MASK_np, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    sample_token = datainfo["token"]
    merged_pcd = o3d.geometry.PointCloud()
    for cam_idx in range(6):
        rgb_image_np = np.asarray(rgb_t_list[cam_idx])
        depth_image = np.mean(depth_t_list[cam_idx], axis=-1) * 100. 
        depth_image =cv2.resize(depth_image, (pcd_width, pcd_height), interpolation=cv2.INTER_CUBIC)
        rgb_image_np = cv2.resize(rgb_image_np, (pcd_width, pcd_height), interpolation=cv2.INTER_CUBIC)
        cam_intrinsics = datainfo['camera_intrinsics'][cam_idx].copy()
        cam2global = datainfo['camera2global'][cam_idx]

        cam2ego = datainfo['camera2ego'][cam_idx]
        ego2global = cam2global @ np.linalg.inv(cam2ego)
        cam2ego_mod = cam2ego.copy()
        # cam2ego_mod[0,3] = cam2ego[0,3]+2  #mod cam2ego
        cam2ego_mod[1,3] = cam2ego[1,3]+mod_y_d  #mod cam2ego
        cam2gloabl_mod = ego2global @ cam2ego_mod 

        cam_intrinsics_rz = resize_cam_intrinsic([ori_width,ori_height],[pcd_width, pcd_height],cam_intrinsics)
        pcd = create_point_cloud_with_open3d(rgb_image_np,depth_image,cam_intrinsics_rz,cam2gloabl_mod)
        merged_pcd += pcd

    for cam_idx in range(6):
        merged_pcd_cam = deepcopy(merged_pcd)
        cam_intrinsics = datainfo['camera_intrinsics'][cam_idx].copy()
        cam_extrinsics = datainfo['camera2global'][cam_idx]
        cam_intrinsics_rz = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam_intrinsics)
        merged_pcd_cam.transform(np.linalg.inv(cam_extrinsics))
        target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics_rz, new_width, new_height)
        
        target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
        # save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
        save_pesudo_cam_path = f"{save_pesudo_root}/{vis_cameras[cam_idx]}"

        
        # os.makedirs(save_gt_cam_path,exist_ok=True)
        # if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
        #     rgb_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")
        # if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.npz"):
        #     np.savez(f"{save_gt_cam_path}/{sample_token}.npz",depth_gt=depth_image)

        os.makedirs(save_pesudo_cam_path,exist_ok=True)
        target_rgb_vis.save(f"{save_pesudo_cam_path}/{sample_token}.jpg")
        depth_vis = depth2color(target_depth,100)
        depth_vis.save(f"{save_pesudo_cam_path}/{sample_token}_depthvis.jpg")
        np.savez(f"{save_pesudo_cam_path}/{sample_token}.npz",depth_proj=target_depth)


    # return merged_pcd

def get_merged_pcd(datainfo1,img_wh):
    # vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = img_wh[0],img_wh[1]
    ori_width,ori_height = 1600,900

    CAM_BACK_MASK = Image.open("../../misc/CAM_BACK_mask.png")
    CAM_BACK_MASK_np = np.asarray(CAM_BACK_MASK)
    CAM_BACK_MASK_np = cv2.resize(CAM_BACK_MASK_np, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    merged_pcd = o3d.geometry.PointCloud()
    rgb1_image_list = []
    depth1_image_list = []


    # rgb_images = []
    # brightness_values = []
    # for cam_idx in range(6): # brightness adjust
    #     image1_path = datainfo1["image_paths"][cam_idx]
    #     image1_path = image1_path.replace("../data/nuscenes","/data/code/MagicDriveDiT/data/nuscenes")
    #     img = Image.open(image1_path).convert("RGB")
    #     rgb_images.append(img)
    #     gray_img = img.convert("L")
    #     stat = ImageStat.Stat(gray_img)
    #     avg_brightness = np.mean(stat.mean)  # 计算平均亮度
    #     brightness_values.append(avg_brightness)
    # target_brightness = np.mean(brightness_values)

    for cam_idx in range(6):
        cam1_intrinsics = datainfo1['camera_intrinsics'][cam_idx]
        cam1_extrinsics = datainfo1['camera2global'][cam_idx]
        cam1_intrinsics = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam1_intrinsics)

        image1_path = datainfo1["image_paths"][cam_idx]
        image1_path = image1_path.replace("../data/nuscenes","/data/nuscenes")
        rgb1_image = Image.open(image1_path)
        rgb1_image = rgb1_image.resize((new_width,new_height))

        # brightness = brightness_values[cam_idx]
        # rgb1_image = rgb_images[cam_idx]
        # brightness_factor = target_brightness / brightness
        # enhancer = ImageEnhance.Brightness(rgb1_image)
        # adjusted_img = enhancer.enhance(brightness_factor)
        # rgb1_image = adjusted_img.resize((new_width,new_height))

        rgb1_image_np = np.asarray(rgb1_image)

        depth1_image =np.load(datainfo1["refine_depth_path"][cam_idx])['depth_pred'].astype(np.float32)
        depth1_image = cv2.resize(depth1_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        Semantic = np.load(datainfo1["semantic_path"][cam_idx])['sem']
        Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        depth1_image[Semantic==27]=100

        # depth1_image[Semantic==55]-=0 #move car pc
        # depth1_image[Semantic==55]=0 #move car pc
        if cam_idx == 4: # CAM_BACK
            depth1_image[CAM_BACK_MASK_np[:,:,0]==0] = 0
        pcd = create_point_cloud_with_open3d(rgb1_image_np,depth1_image,cam1_intrinsics,cam1_extrinsics)
        merged_pcd += pcd

    return merged_pcd
        # rgb1_image_list.append(rgb1_image)
        # depth1_image_list.append(depth1_image)


def resize_cam_intrinsic(ori_wh,resize_wh,cam_intrinsics):
    
    cam_intrinsics[0] = (resize_wh[0]/ori_wh[0])*cam_intrinsics[0]
    cam_intrinsics[1] = (resize_wh[1]/ori_wh[1])*cam_intrinsics[1]
    return cam_intrinsics

  
def proj_frame1_rgbd2_frame2_save(datainfo1,datainfo2,save_gt_root,save_pesudo_root,save_gt=True):

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    merged_pcd = get_merged_pcd(datainfo1,(new_width, new_height))
    sample_token = datainfo2["token"]
    # rgb2_image_list = []
    # depth2_image_list = []
    # target_proj_rgb_list = []
    for cam_idx in range(6):
        merged_pcd_cam = deepcopy(merged_pcd)
        cam2_intrinsics = datainfo2['camera_intrinsics'][cam_idx]
        cam2_extrinsics = datainfo2['camera2global'][cam_idx]
        cam2_intrinsics = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam2_intrinsics)

        image2_path = datainfo2["image_paths"][cam_idx]
        image2_path = image2_path.replace("../data/nuscenes","/data/nuscenes")
        rgb2_image = Image.open(image2_path)
        rgb2_image = rgb2_image.resize((new_width,new_height))
        depth2_image =np.load(datainfo2["refine_depth_path"][cam_idx])['depth_pred'].astype(np.float32)
        depth2_image = cv2.resize(depth2_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        Semantic = np.load(datainfo2["semantic_path"][cam_idx])['sem']
        Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        depth2_image[Semantic==27]=100

        cam2_extrinsics[0,3] = cam2_extrinsics[0,3]+2 #movecam
        merged_pcd_cam.transform(np.linalg.inv(cam2_extrinsics))
        target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam2_intrinsics, new_width, new_height)

        target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
        
        save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
        save_pesudo_cam_path = f"{save_pesudo_root}/{vis_cameras[cam_idx]}"

        if save_gt:
            os.makedirs(save_gt_cam_path,exist_ok=True)
            if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
                rgb2_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")
        os.makedirs(save_pesudo_cam_path,exist_ok=True)
        target_rgb_vis.save(f"{save_pesudo_cam_path}/{sample_token}.jpg")
        # rgb2_image_list.append(rgb2_image)
        # depth2_image_list.append(depth2_image)
        
        # vis_w, vis_h = new_width, new_height
        # new_image = Image.new('RGB', (vis_w, vis_h*2))
        # new_image.paste(rgb2_image,(0, 0))
        # new_image.paste(target_rgb_vis,(0, vis_h))
    
        # target_proj_rgb_list.append(new_image)

        '''
        target_depth[target_depth==0]=100
        target_depth_vis = depth2color(target_depth,clip_max=100)
        ori1_depth_vis = depth2color(depth1_image_list[cam_idx],clip_max=100)
        ori2_depth_vis = depth2color(depth2_image,clip_max=100)
        new_image = Image.new('RGB', (vis_w*2, vis_h*3))
        new_image.paste(rgb1_image_list[cam_idx], (0, 0))  
        new_image.paste(ori1_depth_vis, (vis_w, 0))  
        new_image.paste(rgb2_image, (0, vis_h))
        new_image.paste(ori2_depth_vis, (vis_w, vis_h))
        new_image.paste(target_rgb_vis, (0, vis_h*2))
        new_image.paste(target_depth_vis, (vis_w, vis_h*2))
        new_image.save(f"{save_root}/merge_rgb_proj_{vis_cameras[cam_idx]}{indice}_{interval}.png")
        '''

def proj_movecam_frame1(datainfo,mod_y_d=2):

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    merged_pcd = get_merged_pcd(deepcopy(datainfo),(1200, 675))
    
    
    
    sample_token = datainfo["token"]

    # return None
    # rgb2_image_list = []
    # depth2_image_list = []
    target_proj_rgb_list = []
    target_proj_depth_list = []
    for cam_idx in range(6):
        merged_pcd_cam = deepcopy(merged_pcd)
        cam_intrinsics = datainfo['camera_intrinsics'][cam_idx]
        cam_intrinsics = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam_intrinsics)

        cam2global = datainfo['camera2global'][cam_idx]
        cam2ego = datainfo['camera2ego'][cam_idx]
        ego2global = cam2global @ np.linalg.inv(cam2ego)
        cam2ego_mod = cam2ego.copy()
        # cam2ego_mod[0,3] = cam2ego[0,3]+2  #mod cam2ego
        cam2ego_mod[1,3] = cam2ego[1,3]+mod_y_d  #mod cam2ego
        cam2gloabl_mod = ego2global @ cam2ego_mod 

        # image_path = datainfo["image_paths"][cam_idx]
        # image_path = image_path.replace("../data/nuscenes","/data/code/MagicDriveDiT/data/nuscenes")
        # rgb_image = Image.open(image_path)
        # rgb_image = rgb_image.resize((new_width,new_height))
        # depth_image =np.load(datainfo["refine_depth_path"][cam_idx])['depth_pred'].astype(np.float32)
        # depth_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # Semantic = np.load(datainfo["semantic_path"][cam_idx])['sem']
        # Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # depth_image[Semantic==27]=100

        # cam2_extrinsics[0,3] = cam2_extrinsics[0,3]+2 #movecam
        merged_pcd_cam.transform(np.linalg.inv(cam2gloabl_mod))
        target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, new_width, new_height)
        target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
        target_proj_rgb_list.append(target_rgb_vis)
        target_proj_depth_list.append(target_depth)

    return target_proj_rgb_list,target_proj_depth_list
        # target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
        # save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
        # save_pesudo_cam_path = f"{save_pesudo_root}/{vis_cameras[cam_idx]}"

        # if save_gt:
        #     os.makedirs(save_gt_cam_path,exist_ok=True)
        #     if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
        #         rgb_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")
        #     if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.npz"):
        #         np.savez(f"{save_gt_cam_path}/{sample_token}.npz",depth_gt=depth_image)

        # os.makedirs(save_pesudo_cam_path,exist_ok=True)
        # target_rgb_vis.save(f"{save_pesudo_cam_path}/{sample_token}.jpg")
        # np.savez(f"{save_pesudo_cam_path}/{sample_token}.npz",depth_proj=target_depth)

        # rgb2_image_list.append(rgb2_image)
        # depth2_image_list.append(depth2_image)
        
        # vis_w, vis_h = new_width, new_height
        # new_image = Image.new('RGB', (vis_w, vis_h*2))
        # new_image.paste(rgb2_image,(0, 0))
        # new_image.paste(target_rgb_vis,(0, vis_h))
    
        # target_proj_rgb_list.append(new_image)

        

def rotation_matrix_y(angle):
    theta = np.deg2rad(angle)  # 将角度转换为弧度
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],  # 注意：绕Y轴旋转的矩阵形式
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def proj_FixFrame2video(datainfo,save_pesudo_root=None,move_flag="T",mody=0.08,modx=0.08,Tframe=30):

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    merged_pcd = get_merged_pcd(deepcopy(datainfo),(1200, 675))
    sample_token = datainfo["token"]
    # rgb2_image_list = []
    # depth2_image_list = []
    
    target_proj_rgb_list = [] #cam t
    target_proj_depth_list = []

    for cam_idx in range(6):
        cam_intrinsics = datainfo['camera_intrinsics'][cam_idx]
        cam_intrinsics = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam_intrinsics)

        image1_path = datainfo["image_paths"][cam_idx]
        image1_path = image1_path.replace("../data/nuscenes","/data/nuscenes")
        rgb1_image = Image.open(image1_path)
        rgb1_image = rgb1_image.resize((new_width,new_height))

        depth1_image =np.load(datainfo["refine_depth_path"][cam_idx])['depth_pred'].astype(np.float32)
        depth1_image = cv2.resize(depth1_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        Semantic = np.load(datainfo["semantic_path"][cam_idx])['sem']
        Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        depth1_image[Semantic==27]=100

        target_proj_rgb_cam_list=[rgb1_image]
        target_proj_depth_cam_list=[depth1_image]
        for t in range(Tframe):
            merged_pcd_cam = deepcopy(merged_pcd)
            cam2global = datainfo['camera2global'][cam_idx]
            cam2ego = datainfo['camera2ego'][cam_idx]
            ego2global = cam2global @ np.linalg.inv(cam2ego)
            cam2ego_mod = cam2ego.copy()
            
            
            if move_flag=="T":
                cam2ego_mod[0,3] = cam2ego[0,3]+ t*modx  #mod cam2ego
                cam2ego_mod[1,3] = cam2ego[1,3]+ t*mody  #mod cam2ego
            elif move_flag=="R":
                z_rot=rotation_matrix_y(t*1)
                cam2ego_mod = cam2ego @ z_rot

            cam2gloabl_mod = ego2global @ cam2ego_mod 


            merged_pcd_cam.transform(np.linalg.inv(cam2gloabl_mod))
            target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, new_width, new_height)
            # target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, int(new_width/2), int(new_height/2))
            target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
            
            # save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
            # save_pesudo_sample_cam_path = f"{save_pesudo_root}/{sample_token}/{vis_cameras[cam_idx]}"

            # if save_gt:
            #     os.makedirs(save_gt_cam_path,exist_ok=True)
            #     if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
            #         rgb_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")

            # os.makedirs(save_pesudo_sample_cam_path,exist_ok=True)
            # target_rgb_vis.save(f"{save_pesudo_sample_cam_path}/{t}.jpg")
            # np.savez(f"{save_pesudo_sample_cam_path}/{t}.npz",depth_proj=target_depth)
            target_proj_rgb_cam_list.append(target_rgb_vis)
            target_proj_depth_cam_list.append(target_depth)

        target_proj_rgb_list.append(target_proj_rgb_cam_list)
        target_proj_depth_list.append(target_proj_depth_cam_list)

    return target_proj_rgb_list,target_proj_depth_list

        # rgb2_image_list.append(rgb2_image)
        # depth2_image_list.append(depth2_image)
        
        # vis_w, vis_h = new_width, new_height
        # new_image = Image.new('RGB', (vis_w, vis_h*2))
        # new_image.paste(rgb2_image,(0, 0))
        # new_image.paste(target_rgb_vis,(0, vis_h))
    
        # target_proj_rgb_list.append(new_image)

        

    


def create_point_cloud_with_open3d(rgb_image, depth_image, intrinsics, extrinsics):
    """
    使用 open3d 从单个摄像头生成点云。
    
    参数：
    - rgb_image: RGB 图像 (H, W, 3)
    - depth_image: 深度图像 (H, W)
    - intrinsics: 内参矩阵 (3, 3)
    - extrinsics: 外参矩阵 (4, 4)
    
    返回：
    - pcd: open3d.geometry.PointCloud 对象
    """
    # 创建 open3d 的 RGBD 图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image.astype(np.uint8)),
        o3d.geometry.Image(depth_image.astype(np.float32)),
        depth_scale=1,  # 根据实际深度图的比例设置（通常深度值需要转换为米）
        depth_trunc=120,  # 最大深度裁剪值
        convert_rgb_to_intensity=False
    )

    # 创建 open3d 的内参对象
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(
        width=rgb_image.shape[1],
        height=rgb_image.shape[0],
        fx=intrinsics[0, 0],
        fy=intrinsics[1, 1],
        cx=intrinsics[0, 2],
        cy=intrinsics[1, 2]
    )

    # 从 RGBD 图像和内参生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics
    )

    # 应用外参，将点云从摄像头坐标系转换到世界坐标系
    # transformation = np.linalg.inv(extrinsics)  # 如果外参为从世界到摄像头坐标系，需要取逆
    transformation = extrinsics
    pcd.transform(transformation)

    return pcd

def project_point_cloud_to_image(pcd_ori, intrinsic, width, height):
    """
    将点云投影到图像平面
    :param pcd: Open3D的点云对象
    :param intrinsic: 目标相机的内参矩阵
    :param width: 目标图像的宽度
    :param height: 目标图像的高度
    :return: 投影后的RGB图像和深度图像
    """
    # points = pcd_ori.points
    # colors = pcd_ori.colors

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_ori)

    # o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     width=width, height=height, fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2])
    # # projected_rgbd = o3d.geometry.RGBDImage.create_from_point_cloud(pcd, o3d_intrinsic)
    # pcd = pcd.voxel_down_sample(voxel_size=0.1)
    # camera = o3d.core.Tensor([0, 0, 0], o3d.core.float32)
    # _, pt_map = pcd.hidden_point_removal(camera, 100000)
    # pcd = pcd.select_by_index(pt_map)

    o3d_intrinsic = o3d.core.Tensor(intrinsic[:3, :3])
    rgbd_reproj = pcd.project_to_rgbd_image(width,
                                            height,
                                            o3d_intrinsic,
                                            depth_scale=1,
                                            depth_max=120)

    return np.asarray(rgbd_reproj.color.to_legacy()), np.asarray(rgbd_reproj.depth.to_legacy())




