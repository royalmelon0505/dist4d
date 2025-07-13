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


Rdepth_root = "./dist4d/depth_process/nus_Rdepth"



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
    for cam_idx in range(6):
        cam1_intrinsics = datainfo1['camera_intrinsics'][cam_idx]
        cam1_extrinsics = datainfo1['camera2global'][cam_idx]
        cam1_intrinsics = resize_cam_intrinsic([ori_width,ori_height],[new_width, new_height],cam1_intrinsics)

        image1_path = datainfo1["image_paths"][cam_idx]
        image1_path = image1_path.replace("../data/nuscenes","/data/nuscenes")
        rgb1_image = Image.open(image1_path)
        rgb1_image = rgb1_image.resize((new_width,new_height))
        rgb1_image_np = np.asarray(rgb1_image)

        depth1_image =np.load(datainfo1["refine_depth_path"][cam_idx])['depth_pred'].astype(np.float32)
        depth1_image = cv2.resize(depth1_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        Semantic = np.load(datainfo1["semantic_path"][cam_idx])['sem']
        Semantic = cv2.resize(Semantic, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        depth1_image[Semantic==27]=100
        if cam_idx == 4: # CAM_BACK
            depth1_image[CAM_BACK_MASK_np[:,:,0]==0] = 0

        # depth1_image_list.append(depth1_image)
        pcd = create_point_cloud_with_open3d(rgb1_image_np,depth1_image,cam1_intrinsics,cam1_extrinsics)
        merged_pcd += pcd

    return merged_pcd
        # rgb1_image_list.append(rgb1_image)

def resize_cam_intrinsic(ori_wh,resize_wh,cam_intrinsics):
    
    cam_intrinsics[0] = (resize_wh[0]/ori_wh[0])*cam_intrinsics[0]
    cam_intrinsics[1] = (resize_wh[1]/ori_wh[1])*cam_intrinsics[1]
    return cam_intrinsics

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

def project_point_cloud_to_image(pcd_ori, intrinsic, width, height,downs_flag=False):
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

    if downs_flag:
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
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