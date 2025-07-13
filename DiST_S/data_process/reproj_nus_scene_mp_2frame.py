import os
import json
import numpy as np
import torch
from PIL import Image
# from tqdm.auto import tqdm
from tqdm import tqdm
import cv2
from scipy.ndimage import distance_transform_edt

import pickle

from copy import deepcopy
from tqdm import tqdm
import argparse

from reproj_utils import *


    
def proj_frame1_rgbd2_frame2_save(datainfo1,datainfo2,save_gt_root,save_pesudo_root,save_gt=True):

    vis_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']

    new_width, new_height = 768,432
    ori_width,ori_height = 1600,900

    # merged_pcd = get_merged_pcd(datainfo1,(ori_width/2, ori_height/2))
    merged_pcd = get_merged_pcd(datainfo1,(1200, 675))
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

        merged_pcd_cam.transform(np.linalg.inv(cam2_extrinsics))
        target_rgb, target_depth = project_point_cloud_to_image(merged_pcd_cam, cam2_intrinsics, new_width, new_height)

        target_rgb_vis = Image.fromarray(np.uint8(target_rgb*255))
        
        save_gt_cam_path = f"{save_gt_root}/{vis_cameras[cam_idx]}"
        save_pesudo_cam_path = f"{save_pesudo_root}/{vis_cameras[cam_idx]}"

        if save_gt:
            os.makedirs(save_gt_cam_path,exist_ok=True)
            if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.jpg"):
                rgb2_image.save(f"{save_gt_cam_path}/{sample_token}.jpg")
            if not os.path.exists(f"{save_gt_cam_path}/{sample_token}.npz"):
                np.savez(f"{save_gt_cam_path}/{sample_token}.npz",depth_gt=depth2_image)

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
        "--is_trainset",
        action="store_true",
    )
    parser.add_argument(
        "--reproj_root", type=str,default= "./dist4d/DiST_S/reproj_nus",help="Output directory."
    )
    args = parser.parse_args()

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
    scene_idx=args.scene_idx
    sample_idxs = clip_infos[scene_idx]

    sample_infos =[]
    for idx in sample_idxs:
        sample_info = get_one_data(data_infos,sample_token2depth,idx)
        sample_infos.append(sample_info)

    interval = 2

    image_6cam_list = []
    scene_frame_len = len(sample_idxs)
    save_gt_root = f"{args.reproj_root}/{split_set}/frame_gt/{scene_idx}"

    save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_pesudo+{interval}/{scene_idx}"
    # save_pesudo_root = f"{args.reproj_root}/{split_set}/frame_pesudo-{interval}-d01/{scene_idx}"
    sample_pesudo_list=[]
    for i in tqdm(range(0,scene_frame_len-interval,2)):
        proj_frame1_rgbd2_frame2_save( deepcopy(sample_infos[i]),deepcopy(sample_infos[i+interval]),save_gt_root,save_pesudo_root)
        sample_pesudo_list.append(sample_infos[i+interval]["token"])

    # for i in tqdm(range(interval,scene_frame_len-interval,interval)):
    #     proj_frame1_rgbd2_frame2_save( deepcopy(sample_infos[i]),deepcopy(sample_infos[i-interval]),save_gt_root,save_pesudo_root)
    #     sample_pesudo_list.append(sample_infos[i-interval]["token"])

        # image_6cam_list.append(vis_indice_proj)
    with open(f"{save_pesudo_root}/pesudo_sampletoken.json",'w') as file:
        json.dump(sample_pesudo_list,file)
