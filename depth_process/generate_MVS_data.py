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

from mvs_utils import to_matrix4x4_2,save_MVS_cam_txt,read_MVS_pcd,pair_from_front_id






def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):
    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


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


def main(nusc, indice, nuscenesyaml, args, config):
    # save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']
    

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
        
    
    '''
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
    '''
        
    i = 0
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    # cameras = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    MVS_input_root =f"{args.mvs_input_root}/MVS_Scene{indice}_input/"
    v_range = 100
    i_inter = 1
    dsr = 0#3
    nbp = 0#5
    rad = 0#0.1
    

    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        if i >= len(dict_list):
            print('finish scene!')

            for cam in cameras:
                pair_from_front_id(f"{MVS_input_root}/{cam}/images",f"{MVS_input_root}/{cam}/pair.txt")
            return
        dict = dict_list[i]
        is_key_frame = dict['is_key_frame']
        if not is_key_frame:  # only use key frame as GT
            i = i + 1
            continue
        
        start_index = max(0, i - 2)
        end_index = min(len(dict_list), i + 3)
        # dice = list(range(start_index, end_index))
        # dice.append(i+80)
        # selected_dist_list= dict_list[dice]
        selected_dist_list= dict_list[start_index:end_index]
        # selected_dist_list.append(dict_list[i+40])
       
        cur_dict = dict_list[i]
        cur_lidar_ego_pose = cur_dict["lidar_ego_pose"]
        cur_lidar_ego_T = np.array(cur_lidar_ego_pose['translation'])

        if i>1 and np.linalg.norm(cur_lidar_ego_T - pre_lidar_ego_T) < 0.2:
            i = i + i_inter
            continue
        pre_lidar_ego_T = cur_lidar_ego_T
        print(i)
        # print(ego_to_camera)


        ################## get cam intrinsic and extrinsic ##############
        sample = nusc.get('sample', dict['sample_token'])
        cam_dict = get_cams_img_Tmatrix(nusc,sample)

        
        for cam in cameras:
            cam_info = cam_dict[cam]
            
            
            cam_intrinsic = cam_info["cam_calibrated_sensor"]['camera_intrinsic']
            ego_pose = cam_info["cam_ego_pose"]
            camera_sensor = cam_info["cam_calibrated_sensor"]
            global_to_ego  = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))
            # print(global_to_ego)
            ego_to_camera  = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix, np.array(camera_sensor['translation']))
            # print(ego_to_camera)
            cam_extrinsic = ego_to_camera @ global_to_ego 

            

            cam_img_path=cam_info["image_file"]
            cam_img=Image.open(cam_img_path)

           


        
            ### use for MVS
            depth_num=192
            depth_min=2
            deptm_max=100
            interval_scale = 1.0
            depth_interval = (deptm_max - depth_min) / (depth_num - 1) / interval_scale
            depth_ranges=[depth_min,depth_interval,depth_num,deptm_max]

        
            MVS_input_root_cam = os.path.join(MVS_input_root,cam)
            os.makedirs(os.path.join(MVS_input_root_cam,"images"),exist_ok=True)
            os.makedirs(os.path.join(MVS_input_root_cam,"cams"),exist_ok=True)    
              
            cam_img.save(os.path.join(MVS_input_root_cam,"images",'%08d.jpg' % i))

            # cam_dict_path = os.path.join(MVS_input_root,"cam_dict",f"{i}.pkl")
            # with open(cam_dict_path, 'wb') as f:
            #     pickle.dump(cam_dict, f)
            save_MVS_cam_txt(os.path.join(MVS_input_root_cam,"cams"),i,np.asarray(cam_intrinsic),cam_extrinsic,depth_ranges)
        


        ### save cams info
        # cam_dict_path = os.path.join(vis_root,f"cam_dict_{i}.pkl")
        # with open(cam_dict_path, 'wb') as f:
        #     pickle.dump(cam_dict, f)

        
        i = i + i_inter

        torch.cuda.empty_cache()

        continue
    
        

def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == '__main__':

    parse = ArgumentParser()
    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--config_path', type=str, default='config.yaml')
    parse.add_argument('--mvs_input_root', type=str, default='./mvs_input_root')
    parse.add_argument('--dataroot', type=str, default='/data/nuscenes')
    parse.add_argument('--label_mapping', type=str, default='nuscenes.yaml')
    parse.add_argument('--index', type=int, default=0)

    args = parse.parse_args()

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

    index = args.index

    main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)

    # for index in index_list:
    #     print('processing sequecne:', index)
    #     try:
    #         main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)
    #     except Exception as e:
    #         with open("./scene_error_nksr.txt", 'a') as f:
    #             f.write(str(index) + '\n')
    #         print(e)
    #         continue

    #     with open("./scene_nksr.txt", 'a') as f:
    #         f.write(str(index) + '\n')
