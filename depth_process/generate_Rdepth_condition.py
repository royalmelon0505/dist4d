import torch
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
# from IPython import embed
from tqdm import tqdm
import pickle
# import nksr

import os
import sys
import pdb
import time
import yaml
import torch
# import chamfer
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation
# from megfile import smart_makedirs, smart_path_join, smart_listdir, smart_exists, smart_open

import open3d as o3d
from copy import deepcopy
from argparse import ArgumentParser


import matplotlib.pyplot as plt
from PIL import Image

from multiprocessing import Pool
from functools import partial
import math
import logging
from mvs_utils import to_matrix4x4_2,save_MVS_cam_txt,read_MVS_pcd,pair_from_front_id







def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
        pcd,
        max_nn=20,
        normals=None,
):
    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def lidar_to_world_to_lidar(pc, lidar_calibrated_sensor, lidar_ego_pose,
                            cam_calibrated_sensor,
                            cam_ego_pose):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc

def get_cams_img_Tmatrix(nuscenes,sample):
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    cam_dict={}
    for camera in cameras:
        camera_sample_data = nuscenes.get('sample_data', sample['data'][camera])
        image_file         = os.path.join(nuscenes.dataroot, camera_sample_data["filename"])
        cam_ego_pose       = nuscenes.get('ego_pose', camera_sample_data['ego_pose_token'])
        # global_to_ego      = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))
        cam_calibrated_sensor    = nuscenes.get('calibrated_sensor', camera_sample_data['calibrated_sensor_token']) 
        # camera_intrinsic = cam_calibrated_sensor['camera_intrinsic']
        # ego_to_camera    = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix, np.array(camera_sensor['translation']))

        # global_to_image  = camera_intrinsic @ ego_to_camera @ global_to_ego 
        # tingfeng7 根据相机中得到的信息得到的三个矩阵，将global坐标系的坐标转换到image上
        # image = cv2.imread(image_file) 
        cam_info={"image_file":image_file,
                  "cam_ego_pose":cam_ego_pose,
                  "cam_calibrated_sensor":cam_calibrated_sensor}
        cam_dict[camera]=cam_info
    return cam_dict

def mp_gen(params):
    # if i >= len(dict_list):
    #         print('finish scene!')
    #         return
    nusc,dict_list, mvs_pcd_np, use_mvs_pcd,lidar_calibrated_sensor0, lidar_ego_pose0,object_token_zoo,object_points_vertice,\
        vis_dl_root,save_folder_name,cameras,v_range,list_idx, p_idx = params
    # print(p_idx)
    save_depth_vis = False
    for i in tqdm(list_idx, desc=f"generating processID-{p_idx}" ):
        dict = dict_list[i]
        # is_key_frame = dict['is_key_frame']
        # if not is_key_frame:  # only use key frame as GT
        #     i = i + 1
        #     continue
        # print(i)
        # start_index = max(0, i - 2)
        # end_index = min(len(dict_list), i + 3)
        start_index = max(0, i - 1)
        end_index = min(len(dict_list), i + 2)

        # dice = list(range(start_index, end_index))
        # dice.append(i+80)
        # selected_dist_list= dict_list[dice]
        selected_dist_list= dict_list[start_index:end_index]
        # selected_dist_list.append(dict_list[i+40])
        lidar_pc_list = [dict['lidar_pc'] for dict in selected_dist_list]
        lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

        # lidar_pc = np.array(dict['lidar_pc']).T
        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
        lidar_ego_pose = dict['lidar_ego_pose']

        global_to_Lego  = to_matrix4x4_2(Quaternion(lidar_ego_pose['rotation']).rotation_matrix, np.array(lidar_ego_pose['translation']))
        Lego_to_lidar  = to_matrix4x4_2(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix, np.array(lidar_calibrated_sensor['translation']))
        global_to_lidar = Lego_to_lidar @ global_to_Lego 
        
        if use_mvs_pcd:
            vr_ratio = 1
            Lidar_crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-vr_ratio*v_range, -vr_ratio*v_range, -1], max_bound=[vr_ratio*v_range, vr_ratio*v_range, 50]) #problem
            # mvs_pcd_lidar = deepcopy(mvs_pcd)
            mvs_pcd_lidar = o3d.geometry.PointCloud()
            mvs_pcd_lidar.points = o3d.utility.Vector3dVector(mvs_pcd_np[:, :3])
            mvs_pcd_lidar.transform(global_to_lidar)
            mvs_pcd_lidar = mvs_pcd_lidar.crop(Lidar_crop_bbox)
            # o3d.io.write_point_cloud(os.path.join(pcd_save_root,f"MVS_pros_{i}_wSem.pcd"), mvs_pcd_lidar)

        # print(ego_to_camera)
        
        
        lidar_pc_i = lidar_to_world_to_lidar(lidar_pc.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        # lidar_pc_i_semantic = lidar_to_world_to_lidar(lidar_pc_with_semantic.copy(),
        #                                               lidar_calibrated_sensor0.copy(),
        #                                               lidar_ego_pose0.copy(),
        #                                               lidar_calibrated_sensor,
        #                                               lidar_ego_pose)
        
        point_cloud = lidar_pc_i.points.T[:, :3]
        print("point_cloud shape:",point_cloud.shape)
        # point_cloud_with_semantic = lidar_pc_i_semantic.points.T


        ################# load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################# bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        for j, object_token in enumerate(dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo:
                    points = object_points_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    if points.shape[0] >= 5:
                        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                              torch.from_numpy(gt_bbox_3d[j:j + 1][np.newaxis, :]))
                        points = points[points_in_boxes[0, :, 0].bool()]

                    object_points_list.append(points)
                    # semantics = np.ones_like(points[:, 0:1]) * object_semantic[k]
                    # object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try:  # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            scene_points = np.concatenate([point_cloud, temp])
        except:
            scene_points = point_cloud

        # try:
        #     temp = np.concatenate(object_semantic_list)
        #     scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
        # except:
        #     scene_semantic_points = point_cloud_with_semantic

        ################## remain points with a spatial range ##############
        # print("scene_points shape",scene_points.shape)
        mask = (np.abs(scene_points[:, 0]) < v_range) & (np.abs(scene_points[:, 1]) < v_range) \
               & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 5.0)
        scene_points = scene_points[mask]
        scene_points = np.concatenate((scene_points, np.ones((len(scene_points), 1))), axis=1)
        print("scene_points shape",scene_points.shape)
        ################## get mesh via Possion Surface Reconstruction ##############
        point_cloud_original = o3d.geometry.PointCloud()
        with_normal2 = o3d.geometry.PointCloud()
        point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])

        point_cloud_original = point_cloud_original.voxel_down_sample(voxel_size = 0.1)

        # with_normal = preprocess(point_cloud_original, config)
        # with_normal2.points = with_normal.points
        # with_normal2.normals = with_normal.normals

        # point = np.asarray(with_normal.points)
        # normal = np.asarray(with_normal.normals)

        # point = torch.from_numpy(point).float().cuda()
        # normal = torch.from_numpy(normal).float().cuda()

        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        ################## save original and processed pcd ##############
        

        # o3d.io.write_point_cloud(os.path.join(pcd_save_root,f"point_cloud_SdO_{i}.pcd"), point_cloud_original)
        
        

        # pdb.set_trace()
        proc_pc = point_cloud_original
        proc_pc_points = np.asarray(proc_pc.points)
        scene_points = np.concatenate((proc_pc_points, np.ones((len(proc_pc_points), 1))), axis=1)
        
        ################## get cam intrinsic and extrinsic ##############
        sample = nusc.get('sample', dict['sample_token'])
        cam_dict = get_cams_img_Tmatrix(nusc,sample)

        
        # cameras = ['CAM_FRONT']
        # vis_dl_root = "/data/code/DepthLab/test_cases/nuscene_test_new"
        vis_root = os.path.join(vis_dl_root,save_folder_name,dict['sample_token'])
        os.makedirs(vis_root,exist_ok=True)
        
        depth_vis_root = os.path.join(vis_root,"depth_vis")
        os.makedirs(depth_vis_root,exist_ok=True)

        for cam in cameras:
            cam_info = cam_dict[cam]
            lidar_i_cam = lidar_to_world_to_lidar(scene_points.copy(),
                                                      lidar_calibrated_sensor.copy(),
                                                      lidar_ego_pose.copy(),
                                                      cam_info["cam_calibrated_sensor"],
                                                      cam_info['cam_ego_pose']
                                                      )
            
            
            cam_intrinsic = cam_info["cam_calibrated_sensor"]['camera_intrinsic']
            ego_pose = cam_info["cam_ego_pose"]
            camera_sensor = cam_info["cam_calibrated_sensor"]
            global_to_ego  = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))
            # print(global_to_ego)
            ego_to_camera  = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix, np.array(camera_sensor['translation']))
            # print(ego_to_camera)
            cam_extrinsic = ego_to_camera @ global_to_ego 
            
            

            img_resize_shape=[432,768]
            cam_intrinsic = np.array(cam_intrinsic)*(img_resize_shape[1]/1600)
            intrinsic = o3d.core.Tensor(cam_intrinsic)

            if use_mvs_pcd:
                mvs_pcd_cam = deepcopy(mvs_pcd_lidar)
                trans_lidar_cam = cam_extrinsic @ np.linalg.inv(global_to_lidar)
                mvs_pcd_cam.transform(trans_lidar_cam)
                _,pt_map = mvs_pcd_cam.hidden_point_removal([0,0,0],100000)
                mvs_pcd_cam_hpr = mvs_pcd_cam.select_by_index(pt_map)
                # o3d.io.write_point_cloud(os.path.join("/data/code/uniscene/gs_render/data/MVS_Lidar",f"MVS_pros_{i}_{cam}.pcd"), mvs_pcd_cam_hpr)
                mvs_pcd_cam_fordepth = o3d.t.geometry.PointCloud.from_legacy(mvs_pcd_cam_hpr)

                mvs_depth_reproj = mvs_pcd_cam_fordepth.project_to_depth_image(width=img_resize_shape[1],
                                                height=img_resize_shape[0],
                                                intrinsics=intrinsic,
                                                depth_scale=1,
                                                depth_max=200.0)
                
                mvs_cam_depth = np.asarray(mvs_depth_reproj.to_legacy())

                if save_depth_vis:
                    vis_mvs_cam_depth = mvs_cam_depth.copy()
                    vis_mvs_cam_depth[vis_mvs_cam_depth==0]=200
                    plt.clf()
                    plt.imshow(vis_mvs_cam_depth,cmap='viridis',interpolation='none')
                    plt.savefig(os.path.join(depth_vis_root,f"depth_{i}_{cam}_mvs.png"))
                # np.save(os.path.join(vis_root,f"depth_{i}_{cam}_mvs.npy"),mvs_cam_depth)


            # mvs_pcd_cam_points = np.asarray(proc_pc.points)
            # diameter = np.linalg.norm(
            #     np.asarray(pc_lidar_i_cam_ori.get_max_bound()) - np.asarray(pc_lidar_i_cam_ori.get_min_bound()))
            
            ### hidden point removal and project to depth
            pc_lidar_i_cam_ori = o3d.geometry.PointCloud()
            pc_lidar_i_cam_ori.points = o3d.utility.Vector3dVector(lidar_i_cam.points.T[:,:3])

            _,pt_map = pc_lidar_i_cam_ori.hidden_point_removal([0,0,0],100000)
            pc_lidar_i_cam_ori = pc_lidar_i_cam_ori.select_by_index(pt_map)

            # print(len(pt_map))
            pc_lidar_i_cam = o3d.t.geometry.PointCloud.from_legacy(pc_lidar_i_cam_ori)

            # img_resize_shape=[432,768]
            # cam_intrinsic = np.array(cam_intrinsic)*(img_resize_shape[1]/1600)

            # intrinsic = o3d.core.Tensor(cam_intrinsic)
            depth_reproj = pc_lidar_i_cam.project_to_depth_image(width=img_resize_shape[1],
                                              height=img_resize_shape[0],
                                              intrinsics=intrinsic,
                                              depth_scale=1,
                                              depth_max=200.0)
            
            cam_depth = np.asarray(depth_reproj.to_legacy())

            
            # mvs_pcd_cam += pc_lidar_i_cam_ori
            # o3d.io.write_point_cloud(os.path.join("/data/code/uniscene/gs_render/data/MVS_Lidar",f"MVS+Lidar_{i}_{cam}.pcd"), mvs_pcd_cam)

            ### save depth, vis depth, RGB img

            if save_depth_vis:
                vis_cam_depth = cam_depth.copy()
                vis_cam_depth[vis_cam_depth==0]=200

                plt.clf()
                plt.imshow(vis_cam_depth,cmap='viridis',interpolation='none')
                plt.savefig(os.path.join(depth_vis_root,f"depth_{i}_{cam}.png"))

            cam_img_path=cam_info["image_file"]
            cam_img=Image.open(cam_img_path)
            cam_img_array = np.array(cam_img)
            # cam_img.save(os.path.join(vis_root,f"rgb_{i}_{cam}.png"))
            vis_root_cam_path = os.path.join(vis_root,cam)
            os.makedirs(vis_root_cam_path,exist_ok=True)
            if use_mvs_pcd:
                np.savez_compressed(os.path.join(vis_root_cam_path,f"rgb_depth_{i}.npz"), rgb=cam_img_array, raw_depth=cam_depth,mvs_depth=mvs_cam_depth )
            else:
                np.savez_compressed(os.path.join(vis_root_cam_path,f"rgb_depth_{i}.npz"), rgb=cam_img_array, raw_depth=cam_depth)
            # np.save(os.path.join(vis_root,f"depth_{i}_{cam}.npy"),cam_depth)
            
            
                
            

           
        

def gen_raw_mvs_depth_mp(nusc,dict_list, mvs_pcd_np, use_mvs_pcd,lidar_calibrated_sensor0, lidar_ego_pose0,object_token_zoo,object_points_vertice,\
        vis_dl_root,save_folder_name,cameras,v_range, pros_list, p_num):
    '''
    多线程对当前scan进行处理, 是在单个scan内进行截断加速
    '''
    params_mp = []
    print("mp_gen_raw_mvs_depth")
    pair_inter = math.ceil(len(pros_list) / p_num)
    for p_idx in range(p_num):
        id_s, id_e = p_idx*pair_inter, (p_idx+1)*pair_inter
        id_e = len(pros_list) if id_e >= len(pros_list) else id_e
        params_mp.append([nusc,dict_list, mvs_pcd_np, use_mvs_pcd,lidar_calibrated_sensor0, lidar_ego_pose0,object_token_zoo,object_points_vertice,\
        vis_dl_root,save_folder_name,cameras,v_range,pros_list[id_s:id_e],p_idx])

    partial_func = partial(mp_gen)
    p = Pool(p_num)
    p.map(partial_func, params_mp)
    p.close()
    p.join()

def main(nusc, indice, nuscenesyaml, args, config):
    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    # if args.split == 'train':
    #     if my_scene['token'] in val_list:
    #         return
    # elif args.split == 'val':
    #     if my_scene['token'] not in val_list:
    #         return
    # elif args.split == 'all':
    #     pass
    # else:
    #     raise NotImplementedError

    # load the first sample to start
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # collect LiDAR sequence
    dict_list = []
    sample_token_list = []
    while True:
        ############################# Init: is has lidarseg labels ##########################
        flag_has_lidarseg = True
        try:
            lidar_sd_token = lidar_data['token']
            seg_name = nusc.get('lidarseg', lidar_sd_token)
        except KeyError:
            flag_has_lidarseg = False

        ############################# get boxes ##########################
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
        boxes_token = [box.token for box in boxes]
        object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
        object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]

        ############################# get object categories ##########################
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml['labels']):
                if category == nuscenesyaml['labels'][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # Slightly expand the bbox to wrap all object points
        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename']  # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
        if lidar_data['is_key_frame'] and flag_has_lidarseg:  # only key frame has semantic annotations
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
        object_points_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:, j].bool()
            object_points = pc0[object_points_mask]
            object_points_list.append(object_points)
            j = j + 1

        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        ############################# get point mask of the vehicle itself ##########################
        self_range = config['self_range']
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > self_range[0]) |
                                        (np.abs(pc0[:, 1]) > self_range[1]) |
                                        (np.abs(pc0[:, 2]) > self_range[2]))

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_pc = lidar_to_world_to_lidar(pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(),
                                           lidar_calibrated_sensor0,
                                           lidar_ego_pose0)
        
        

        ################## record Non-key frame information into a dict  ########################
        dict = {"object_tokens": object_tokens,
                "object_points_list": object_points_list,
                "lidar_pc": lidar_pc.points,
                "lidar_ego_pose": lidar_ego_pose,
                "lidar_calibrated_sensor": lidar_calibrated_sensor,
                "lidar_token": lidar_data['token'],
                "sample_token": lidar_data['sample_token'],
                "is_key_frame": lidar_data['is_key_frame'],
                "has_lidarseg": flag_has_lidarseg,
                "gt_bbox_3d": gt_bbox_3d,
                "converted_object_category": converted_object_category,
                "pc_file_name": pc_file_name.split('/')[-1]}
        ################# record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame'] and flag_has_lidarseg:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                             lidar_calibrated_sensor.copy(),
                                                             lidar_ego_pose.copy(),
                                                             lidar_calibrated_sensor0,
                                                             lidar_ego_pose0)
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

        dict_list.append(dict)
        sample_token_list.append(dict['sample_token'])
        ################## go to next frame of the sequence  ########################
        curr_sample_token = lidar_data['sample_token']
        next_sample_token = nusc.get('sample', curr_sample_token)['next']
        if next_sample_token != '':
            next_lidar_token = nusc.get('sample', next_sample_token)['data'][sensor]
            lidar_data = nusc.get('sample_data', next_lidar_token)
        else:
            break
        # next_token = lidar_data['next']
        # if next_token != '':
        #     lidar_data = nusc.get('sample_data', next_token)
        # else:
        #     break

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    for dict in dict_list:
        if dict['is_key_frame'] and dict['has_lidarseg']:
            lidar_pc_with_semantic_list.append(dict['lidar_pc_with_semantic'])
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T

    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i, object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    object_points_dict = {}

    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token:
                    object_points = dict['object_points_list'][i]
                    if object_points.shape[0] > 0:
                        object_points = object_points[:, :3] - dict['gt_bbox_3d'][i][:3]
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        object_points_dict[query_object_token].append(rotated_object_points)
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                axis=0)

    object_points_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_vertice.append(point_cloud[:, :3])
    # print('object finish')
    
    i = 0
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    # cameras = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # MVS_input_root ="/data/code/visualrecon/MVS_Scene1_input/"
    v_range = 100
    i_inter = 1
    dsr = 0#3
    nbp = 0#5
    rad = 0#0.1
    use_mvs_pcd =True
    
    
    if use_mvs_pcd:
        if args.keyframe_pcd:
            pcd_path = f"{args.mvs_input_root}/MVS_Scene{indice}_output_keyframe/merged_6cam.pcd"
            if os.path.exists(pcd_path):
                mvs_pcd = o3d.io.read_point_cloud(pcd_path)
                mvs_pcd=mvs_pcd.uniform_down_sample(every_k_points=2)
                cl, ind = mvs_pcd.remove_radius_outlier(nb_points=5, radius=0.5)
                mvs_pcd = mvs_pcd.select_by_index(ind)
                mvs_pcd=mvs_pcd.voxel_down_sample(voxel_size = 0.2)
                mvs_pcd_points = np.asarray(mvs_pcd.points)
            else:
                use_mvs_pcd = False
                print("No keyframe mvs pcd files! Without using it!")

        else: 
            pcd_path = f"{args.mvs_input_root}/MVS_Scene{indice}_output/merged_6cam.pcd"
            # if not os.path.exists(pcd_path):
            #     pcd_path = f"/mnt/bringup/crosshairs/guojiazhe/MVSpcd_nus/{indice}_output/merged_6cam.pcd"
            if os.path.exists(pcd_path):
                mvs_pcd = o3d.io.read_point_cloud(pcd_path)
                mvs_pcd=mvs_pcd.uniform_down_sample(every_k_points=5)
                cl, ind = mvs_pcd.remove_radius_outlier(nb_points=8, radius=0.5)
                mvs_pcd = mvs_pcd.select_by_index(ind)
                mvs_pcd=mvs_pcd.voxel_down_sample(voxel_size = 0.2)
                mvs_pcd_points = np.asarray(mvs_pcd.points)
            else:
                use_mvs_pcd = False
                # print(e)
                mvs_pcd_points = None
                print("No mvs pcd files! Without using it!")
    
        # o3d.io.write_point_cloud(os.path.join("/data/code/uniscene/gs_render/data/MVS_Lidar",f"MVS_down_{indice}_805.pcd"), mvs_pcd)
        # cl, ind = mvs_pcd.remove_radius_outlier(nb_points=10, radius=1)
        
        # sp_pcd_root = "/data/code/visualrecon/MVS_Scene1_output/sp_pcd_p0.1t3"
        # mvs_pcd = read_MVS_pcd(sp_pcd_root)
        # crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-v_range, -20, 0], max_bound=[v_range, -1, 3*v_range]) #problem
    
    # save_folder_name = f"D2_{indice}_vr{v_range}_{i_inter}_d768"
    save_folder_name = args.save_folder_name
    # save_folder_name = f"{indice}_cur"
    pcd_save_root = os.path.join(save_path,save_folder_name)

    vis_dl_root = save_path#"/data/code/DepthLab/test_cases/nuscene_test_new"
    dl_save_root = os.path.join(vis_dl_root,save_folder_name)
    os.makedirs(dl_save_root,exist_ok=True)
    os.makedirs(pcd_save_root,exist_ok=True)
    with open(os.path.join(dl_save_root,"sample_token_list.json"), 'w') as file:
        json.dump(sample_token_list, file)
    # json.dump(,,indent=4)
    
    max_len = len(dict_list)
    p_num = 2
    pros_list = list(range(0,max_len,i_inter))
    gen_raw_mvs_depth_mp(nusc,dict_list, mvs_pcd_points, use_mvs_pcd,lidar_calibrated_sensor0, lidar_ego_pose0,object_token_zoo,object_points_vertice,\
        vis_dl_root,save_folder_name,cameras,v_range, pros_list, p_num)
    
    # while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        

    #     continue
    
    
        

def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == '__main__':

    parse = ArgumentParser()
    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--config_path', type=str, default='config.yaml')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='/data/nus_Rdepth/')
    parse.add_argument('--dataroot', type=str, default='./data/nuscenes')
    parse.add_argument('--label_mapping', type=str, default='nuscenes.yaml')
    parse.add_argument('--save_folder_name', type=str)
    parse.add_argument('--mvs_input_root', type=str, default='./mvs_input_root')
    parse.add_argument('--index', type=int, default=0)
    parse.add_argument('--keyframe_pcd', action='store_true')

    args = parse.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_start_method('spawn')
    if args.dataset == 'nuscenes':
        nusc = NuScenes(version='advanced_12Hz_trainval',
                        dataroot=args.dataroot,
                        verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
    else:
        print('Dataset not supported')

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)

    # index_list = args.index_list
    index = args.index
    main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)

    # for index in index_list:
    #     print('processing sequecne:', index)
    #     # main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)
    #     logger.info(f"==> Start process lidar depth and mvs depth")
    #     pros_raw_depth_start = time.time()
    #     try:
    #         main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)
    #     except Exception as e:
    #         with open("./scene_error_nksr.txt", 'a') as f:
    #             f.write(str(index) + '\n')
    #         print(e)
    #         continue

    #     with open("./scene_nksr.txt", 'a') as f:
    #         f.write(str(index) + '\n')

    #     logger.info(f"==> Finish data-process using {time.time() - pros_raw_depth_start} s.")