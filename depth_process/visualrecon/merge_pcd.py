import open3d
import os
import argparse
import numpy as np
import math
import time
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def get_point_cloud(path):
    ply = open3d.io.read_point_cloud(path)
    print(ply)
    return ply

def merge_ply(src_path, save_path, 
              ds=True, ds_cls='voxel', voxel_size=0.05,
              split_num=4):
    '''
    分割方法: 从fuse好的ply中导出point, 根据pts的顺序进行划分, 问题在于ply中的点竟然是不带顺序的
    当前问题: voxel_down_sample是前后视单独做的, 并不合理, 应当前后视拼到一起之后再做down_sample
    '''
    content_list = sorted(os.listdir(src_path))
    src_ply = [c for c in content_list if c.endswith('.ply')]
    if len(src_ply) <= 1:
        print(f"Only {len(src_ply)} pcd in given path")
        return
    
    os.makedirs(save_path, exist_ok=True)
    tgt_pts = [[] for _ in range(split_num)]
    tgt_colors = [[] for _ in range(split_num)]

    for ply in src_ply:
        start_time = time.time()
        print(f"downsample and split {ply} ....")
        pcd = get_point_cloud(os.path.join(src_path, ply))
        if ds:
            if ds_cls == 'uniform':
                pcd = pcd.uniform_down_sample(every_k_points=40)
            elif ds_cls == 'voxel':
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            else:
                raise NotImplementedError
        
        pcd_pts = np.array(pcd.points)
        pcd_colors = np.array(pcd.colors)
        for sp_idx in range(split_num):
            sp_inter = math.ceil(len(pcd_pts)/split_num)
            sp_s, sp_e = sp_idx*sp_inter, (sp_idx+1)*sp_inter
            sp_e = len(pcd_pts) if sp_e >= len(pcd_pts) else sp_e
            tgt_pts[sp_idx].append(pcd_pts[sp_s:sp_e])
            tgt_colors[sp_idx].append(pcd_colors[sp_s:sp_e])
        print(f"downsample and split {ply} cost {time.time()-start_time} s.")

    # save sp pcd
    for sp_idx in range(split_num):
        save_file = os.path.join(save_path, f"sp_{sp_idx}.pcd")
        new_pcd = open3d.geometry.PointCloud()
        pts = np.vstack(tgt_pts[sp_idx]).reshape(-1,3)
        colors = np.vstack(tgt_colors[sp_idx]).reshape(-1,3)
        new_pcd.points = open3d.utility.Vector3dVector(pts)
        new_pcd.colors = open3d.utility.Vector3dVector(colors)
        print(f"Save {len(new_pcd.points)} in {save_file.split('/')[-1]}")
        open3d.io.write_point_cloud(save_file, new_pcd)


def merge_ply_v3(src_path, save_path, cam_list=['cam_front_120', 'cam_back_120'],
                ds=True, ds_cls='voxel', voxel_size=0.05, split_num=4):
    '''
    直接读取MVS的out_path, 从保存的各帧npy中merge和分割点云, 可以做到根据img_id精准分割
    '''
    os.makedirs(save_path, exist_ok=True)
    # src_pts, src_colors, = [], []
    src_ids = []
    for cam in cam_list:
        src_ids.append(sorted(os.listdir(os.path.join(src_path, cam, 'pts_npy'))))
    
    for sp_idx in range(split_num):
        save_file = os.path.join(save_path, f"sp_{sp_idx}.pcd")
        start_time = time.time()

        tgt_pts_sp, tgt_colors_sp = [], []
        for cam_idx in range(len(cam_list)):
            sp_inter = math.ceil(len(src_ids[cam_idx])/split_num)
            sp_s, sp_e = sp_idx*sp_inter, (sp_idx+1)*sp_inter
            sp_e = len(src_ids[cam_idx]) if sp_e >= len(src_ids[cam_idx]) else sp_e
            for id in tqdm(range(sp_s, sp_e), desc=f'split-{sp_idx} load {cam_list[cam_idx]}'):
                pt = np.load(os.path.join(src_path, cam_list[cam_idx], 'pts_npy', src_ids[cam_idx][id]))
                color = np.load(os.path.join(src_path, cam_list[cam_idx], 'colors_npy', src_ids[cam_idx][id]))
                tgt_pts_sp.append(pt)
                tgt_colors_sp.append(color)
        
        pts = np.vstack(tgt_pts_sp).reshape(-1,3)
        colors = np.vstack(tgt_colors_sp).reshape(-1,3)

        new_pcd = open3d.geometry.PointCloud()
        new_pcd.points = open3d.utility.Vector3dVector(pts)
        new_pcd.colors = open3d.utility.Vector3dVector(colors)
        
        print(f"split-{sp_idx},  {ds_cls} downample with param {voxel_size}....")
        if ds:
            if ds_cls == 'uniform':
                new_pcd = new_pcd.uniform_down_sample(every_k_points=40)
            elif ds_cls == 'voxel':
                new_pcd = new_pcd.voxel_down_sample(voxel_size=voxel_size)
            else:
                raise NotImplementedError
            
        print(f"save {len(new_pcd.points)} in {save_file.split('/')[-1]}")
        open3d.io.write_point_cloud(save_file, new_pcd)
        print(f"downsample and split cost {time.time()-start_time} s.")
        

def mr_mp(params):
    '''
    多线程分割pcd, 每个线程处理一个split
    '''
    src_path, save_file, cam_list, ds, ds_cls, \
        voxel_size, split_num, sp_idx, src_ids, fuse_sem_color = params

    start_time = time.time()
    tgt_pts_sp, tgt_colors_sp = [], []
    for cam_idx in range(len(cam_list)):
        sp_inter = math.ceil(len(src_ids[cam_idx])/split_num)
        sp_s, sp_e = sp_idx*sp_inter, (sp_idx+1)*sp_inter
        sp_e = len(src_ids[cam_idx]) if sp_e >= len(src_ids[cam_idx]) else sp_e
        for id in tqdm(range(sp_s, sp_e), desc=f'split-{sp_idx} load {cam_list[cam_idx]}'):
            pt = np.load(os.path.join(src_path, cam_list[cam_idx], 'pts_npy', src_ids[cam_idx][id]))
            if not fuse_sem_color:
                color = np.load(os.path.join(src_path, cam_list[cam_idx], 'colors_npy', src_ids[cam_idx][id]))
            else:
                color = np.load(os.path.join(src_path, cam_list[cam_idx], 'sem_colors_npy', src_ids[cam_idx][id]))
            tgt_pts_sp.append(pt)
            tgt_colors_sp.append(color)
    
    pts = np.vstack(tgt_pts_sp).reshape(-1,3)
    colors = np.vstack(tgt_colors_sp).reshape(-1,3)

    new_pcd = open3d.geometry.PointCloud()
    new_pcd.points = open3d.utility.Vector3dVector(pts)
    new_pcd.colors = open3d.utility.Vector3dVector(colors)
    
    print(f"split-{sp_idx},  {ds_cls} downample with param {voxel_size}....")
    if ds:
        if ds_cls == 'uniform':
            new_pcd = new_pcd.uniform_down_sample(every_k_points=40)
        elif ds_cls == 'voxel':
            new_pcd = new_pcd.voxel_down_sample(voxel_size=voxel_size)
        else:
            raise NotImplementedError
        
    print(f"save {len(new_pcd.points)} in {save_file.split('/')[-1]}")
    open3d.io.write_point_cloud(save_file, new_pcd)
    print(f"downsample and split cost {time.time()-start_time} s.")


def merge_ply_v3_mp(src_path, save_path, cam_list=['cam_front_120', 'cam_back_120'],
                ds=True, ds_cls='voxel', voxel_size=0.05, split_num=4, fuse_sem_color=False, **kwargs):
    '''多线程
    直接读取MVS的out_path, 从保存的各帧npy中merge和分割点云, 可以做到根据img_id精准分割
    '''
    # assert split_num >=2, f"invalid num ({split_num}) for split_num"
    os.makedirs(save_path, exist_ok=True)
    src_ids = []
    for cam in cam_list:
        src_ids.append(sorted(os.listdir(os.path.join(src_path, cam, 'pts_npy'))))
    
    params = []
    for sp_idx in range(split_num):
        save_file = os.path.join(save_path, f"sp_{sp_idx}.pcd")
        params.append([src_path, save_file, cam_list, ds, ds_cls, voxel_size,\
                        split_num, sp_idx, src_ids, fuse_sem_color])
    
    partial_func = partial(mr_mp)
    p = Pool(split_num)
    p.map(partial_func, params)
    p.close()
    p.join()
    
    pcd_list = os.listdir(save_path)
    merged_pcd = open3d.geometry.PointCloud()
    for pcd_path in pcd_list:
        pcd = open3d.io.read_point_cloud(os.path.join(save_path,pcd_path))
        merged_pcd+=pcd
    merged_pcd =  merged_pcd.voxel_down_sample(voxel_size=0.1)
    open3d.io.write_point_cloud(os.path.join(src_path,"merged_6cam.pcd"), merged_pcd)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Downsample ply results for visualization.')

    parser.add_argument('--src_path', required=True, help='src data path')
    parser.add_argument('--save_path', required=True, help='save data path')
    parser.add_argument('--fuse_sem_color', action='store_true', help='fuse sem color instead of RGB color')
    parser.add_argument('--ds_cls', default='voxel', help='downsample ration')
    parser.add_argument('--ds_voxel', type=float, default=0.05, help='downsample voxel size')
    parser.add_argument('--split_num', type=int, default=5, help='split number of src point cloud')

    args = parser.parse_args()

    merge_ply_v3_mp(src_path=args.src_path,
              save_path=args.save_path,
              cam_list=['cam_front_120', 'cam_back_120'],
              ds=True,
              ds_cls=args.ds_cls,
              voxel_size=args.ds_voxel,
              split_num=args.split_num,
              fuse_sem_color=args.fuse_sem_color)

