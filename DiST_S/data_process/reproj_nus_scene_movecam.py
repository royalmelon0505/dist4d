import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import cv2
from scipy.ndimage import distance_transform_edt

import pickle
from pyquaternion import Quaternion
import open3d as o3d
from copy import deepcopy
from tqdm import tqdm
import argparse
import sys

from reproj_utils import *



def proj_movecam_frame1_save(datainfo, save_gt_root,save_pesudo_root,save_gt=True,mod_y_d=2):

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    merged_pcd = get_merged_pcd(deepcopy(datainfo),(1200, 675))
    
    
    
    sample_token = datainfo["token"]

    # return None
    # rgb2_image_list = []
    # depth2_image_list = []
    # target_proj_rgb_list = []
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

        image_path = datainfo["image_paths"][cam_idx]
        image_path = image_path.replace("../data/nuscenes","/data/nuscenes")
        rgb_image = Image.open(image_path)
        rgb_image = rgb_image.resize((new_width,new_height))
        depth_image =np.load(datainfo["refine_depth_path"][cam_idx])['depth_pred'].astype(np.float32)
        depth_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        Semantic = np.load(datainfo["semantic_path"][cam_idx])['sem']
        Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        depth_image[Semantic==27]=100

        # cam2_extrinsics[0,3] = cam2_extrinsics[0,3]+2 #movecam
        merged_pcd_cam.transform(np.linalg.inv(cam2gloabl_mod))
        target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, new_width, new_height)

        target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
        
        save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
        save_pesudo_cam_path = f"{save_pesudo_root}/{vis_cameras[cam_idx]}"

        if save_gt:
            os.makedirs(save_gt_cam_path,exist_ok=True)
            if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
                rgb_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")
            if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.npz"):
                np.savez(f"{save_gt_cam_path}/{sample_token}.npz",depth_gt=depth_image)

        os.makedirs(save_pesudo_cam_path,exist_ok=True)
        target_rgb_vis.save(f"{save_pesudo_cam_path}/{sample_token}.jpg")
        np.savez(f"{save_pesudo_cam_path}/{sample_token}.npz",depth_proj=target_depth)
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

def proj_FixFrame2video_save(datainfo,save_pesudo_root,mody=0.1,Tframe=20):

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    merged_pcd = get_merged_pcd(deepcopy(datainfo),(1200, 675))
    sample_token = datainfo["token"]
    # rgb2_image_list = []
    # depth2_image_list = []
    # target_proj_rgb_list = []
    
    for cam_idx in range(6):
        cam_intrinsics = datainfo['camera_intrinsics'][cam_idx]
        cam_intrinsics = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam_intrinsics)
        for t in range(Tframe):
            merged_pcd_cam = deepcopy(merged_pcd)
            cam2global = datainfo['camera2global'][cam_idx]
            cam2ego = datainfo['camera2ego'][cam_idx]
            ego2global = cam2global @ np.linalg.inv(cam2ego)
            cam2ego_mod = cam2ego.copy()
            
            # cam2ego_mod[0,3] = cam2ego[0,3]+2  #mod cam2ego
            cam2ego_mod[1,3] = cam2ego[1,3]+ t*mody  #mod cam2ego

            # z_rot=rotation_matrix_y(t*2)
            # cam2ego_mod = cam2ego @ z_rot

            cam2gloabl_mod = ego2global @ cam2ego_mod 


            merged_pcd_cam.transform(np.linalg.inv(cam2gloabl_mod))
            target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, new_width, new_height)
            # target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam_intrinsics, int(new_width/2), int(new_height/2))
            target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
            
            # save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
            save_pesudo_sample_cam_path = f"{save_pesudo_root}/{sample_token}/{vis_cameras[cam_idx]}"

            # if save_gt:
            #     os.makedirs(save_gt_cam_path,exist_ok=True)
            #     if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
            #         rgb_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")

            os.makedirs(save_pesudo_sample_cam_path,exist_ok=True)
            target_rgb_vis.save(f"{save_pesudo_sample_cam_path}/{t}.jpg")
            np.savez(f"{save_pesudo_sample_cam_path}/{t}.npz",depth_proj=target_depth)
        # rgb2_image_list.append(rgb2_image)
        # depth2_image_list.append(depth2_image)
        
        # vis_w, vis_h = new_width, new_height
        # new_image = Image.new('RGB', (vis_w, vis_h*2))
        # new_image.paste(rgb2_image,(0, 0))
        # new_image.paste(target_rgb_vis,(0, vis_h))
    
        # target_proj_rgb_list.append(new_image)



if "__main__" == __name__:

    parser = argparse.ArgumentParser(
        description="reproj data process nuscenes."
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mod_y_d",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--is_trainset",
        action="store_true",
    )
    parser.add_argument(
        "--reproj_root", type=str,default= "./dist4d/DiST_S/reproj_nus",help="Output directory."
    )
    args = parser.parse_args()

    scene_idx=args.scene_idx

    val_use_dict={"1": "11_scene", "2": "12_scene", "3": "13_scene", "4": "14_scene", "10": "36_scene", "13": "75_scene", "17": "79_scene", "21": "83_scene", "22": "84_scene", "25": "87_scene", "26": "88_scene", "28": "90_scene", "29": "91_scene", "30": "92_scene", "32": "214_scene", "43": "257_scene", "45": "259_scene", "47": "261_scene", "48": "262_scene", "51": "410_scene", "53": "412_scene", "55": "414_scene", "56": "436_scene", "59": "439_scene", "62": "442_scene", "63": "443_scene", "64": "444_scene", "65": "445_scene", "66": "446_scene", "67": "447_scene"}
    if args.is_trainset==False:
        if str(scene_idx) not in val_use_dict.keys():
            print(f"skip {scene_idx}")
            sys.exit() 
    

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

    # import pdb;pdb.set_trace()
    
    sample_idxs = clip_infos[scene_idx]

    sample_infos =[]
    for idx in sample_idxs:
        sample_info = get_one_data(data_infos,sample_token2depth,idx)
        sample_infos.append(sample_info)

    interval=2

    image_6cam_list = []
    scene_frame_len = len(sample_idxs)
    save_gt_root = f"{args.reproj_root}/{split_set}/frame_gt/{scene_idx}"

    # save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_pesudo+{interval}/{scene_idx}"
    
    
    sample_pesudo_list=[]
    # for i in tqdm(range(0,scene_frame_len-interval,interval)):
    #     proj_frame1_rgbd2_frame2_save( deepcopy(sample_infos[i]),deepcopy(sample_infos[i+interval]),save_gt_root,save_pesudo_root)
    #     sample_pesudo_list.append(sample_infos[i+interval]["token"])

    # ----- different sample  move cam ---- 
    
    mod_y_d = args.mod_y_d

    # if mod_y_d>0:
    #     save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_movecam2egoy+{mod_y_d}/{scene_idx}"
    # else:
    #     save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_movecam2egoy{mod_y_d}/{scene_idx}"

    #left
    save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_movecam2egoy+{mod_y_d}/{scene_idx}"
    # if os.path.exists(f"{save_pesudo_root}/pesudo_sampletoken.json"):
    #     sys.exit()
    for i in tqdm(range(0,scene_frame_len-interval,interval)):
        proj_movecam_frame1_save(deepcopy(sample_infos[i]),save_gt_root,save_pesudo_root,mod_y_d=mod_y_d)
        sample_pesudo_list.append(sample_infos[i]["token"])

    with open(f"{save_pesudo_root}/pesudo_sampletoken.json",'w') as file:
        json.dump(sample_pesudo_list,file)
    
    '''
    sample_pesudo_list =[]
    #right
    save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_movecam2egoy-{mod_y_d}/{scene_idx}"
    for i in tqdm(range(0,scene_frame_len-interval,interval)):
        proj_movecam_frame1_save(deepcopy(sample_infos[i]),save_gt_root,save_pesudo_root,mod_y_d=-mod_y_d)
        sample_pesudo_list.append(sample_infos[i]["token"])
        
    with open(f"{save_pesudo_root}/pesudo_sampletoken.json",'w') as file:
        json.dump(sample_pesudo_list,file)
    '''


    # ----  same sample move cam --- 
    # save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_movecam2egoy_onesample/{scene_idx}"
    # proj_FixFrame2video_save(deepcopy(sample_infos[10]),save_pesudo_root)



# save_root = 'vis_warp'
# save_depth_color_video(image_6cam_list,os.path.join(save_root,f"merge_6cam_scene{scene_idx}-{interval}.mp4"),fps=6) 

# import pdb;pdb.set_trace()


