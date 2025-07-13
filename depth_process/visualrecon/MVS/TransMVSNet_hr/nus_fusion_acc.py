import argparse
import os
import sys
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from MVS.TransMVSNet_hr.utils import print_args
from MVS.TransMVSNet_hr.datasets.data_io import read_pfm
from multiprocessing import Pool
from functools import partial



Cityscapes = {
        0: 'road', 1: 'sidewalk', 2:'building', 3: 'wall',
        4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign',
        8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider',
        13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
        }

Mapillary_Vistas = {
        0: 'bird', 1: 'ground animal', 2: 'curb', 3: 'fence', 4: 'guard rail',
        5: 'barrier', 6: 'wall', 7: 'bike lane', 8: 'crosswalk-plain',
        9: 'curb cut', 10: 'parking', 11: 'pedestrian area', 12: 'rail track',
        13: 'road', 14: 'service lane', 15: 'sidewalk', 16: 'bridge', 17: 'building',
        18: 'tunnel', 19: 'person', 20: 'bicyclist', 21: 'motorcyclist', 22: 'other rider',
        23: 'lane marking - crosswalk', 24: 'lane marking - general', 25: 'mountain',
        26: 'sand', 27: 'sky', 28: 'snow', 29: 'terrain', 30: 'vegetation', 31: 'water',
        32: 'banner', 33: 'bench', 34: 'bike rack', 35: 'billboard', 36: 'catch basin',
        37: 'CCTV camera', 38: 'fire hydrant', 39: 'junction box', 40: 'mailbox',
        41: 'manhole', 42: 'phone booth', 43: 'phthole', 44: 'street light', 45: 'pole',
        46: 'traffic sign frame', 47: 'utility pole', 48: 'traffic light', 49: 'traffic sign(back)',
        50: 'traffic sign(front)', 51: 'trash can', 52: 'bicycle', 53: 'boat', 54: 'bus', 55: 'car',
        56: 'caravan', 57: 'motorcycle', 58: 'on rails', 59: 'other vehicle', 60: 'trailer',
        61: 'truck', 62: 'wheeled slow', 63: 'car mount', 64: 'ego vehicle', 65: 'unlabeled'
        }

SAVE_CLS = {
    #'Cityscapes': [0,1,6,7],
    'Cityscapes': [0,1,2,3,4,5,6,7,8,9,],
    # full static class
    'Mapillary_Vistas': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,\
                        23,24,15,26,29,30,31,32,35,36,37,38,39,40,41,42,\
                        44,45,46,47,48,49,50,51]
    # simple static class
    # 'Mapillary_Vistas': [7,8,10,11,13,14,15,16,23,24,44,46,48,50]
}


parser = argparse.ArgumentParser(description='Filter and fuse depth maps')

parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--input_path', help='MVS_input sdata path')
parser.add_argument('--mp_num', type=int, default=12, help='multi-process num')
parser.add_argument('--scale', type=float, default=0.8, help='scale factor to resize the MVS output')
parser.add_argument('--Semantic_path', help='path to semantic results')
parser.add_argument('--Semantic_dilate_size', type=int, default=0, help='kernel size for filating mask')
parser.add_argument('--Semantic_type', default='Mapillary_Vistas', choices=['Mapillary_Vistas', 'Cityscapes'], help='semantic types')
parser.add_argument('--save_sem_color', action='store_true')
parser.add_argument('--testlist', help='testing scan list')
parser.add_argument('--outdir', help='output dir')
parser.add_argument('--photo_threshold', type=float, default=0.3, help='photo threshold for filter confidence')
parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
parser.add_argument('--config', default='none')
parser.add_argument('--indice', default='none')

args = parser.parse_args()
print("argv:", sys.argv[1:])
# print_args(args)


def read_camera_parameters_raw(filename,scale,index,flag):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics[:2, :] *= scale

    if (flag==0):
        intrinsics[0,2]-=index
    else:
        intrinsics[1,2]-=index

    return intrinsics, extrinsics

def cal_ext_dist(ref_extrinsics,src_extrinsics):
    ref_e_T = ref_extrinsics[0:3,3]
    src_e_T = src_extrinsics[0:3,3]
    # print(src_e_T)
    return np.linalg.norm(ref_e_T - src_e_T)

def read_camera_parameters(filename, scale_w, scale_h):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    intrinsics[0, :] *= scale_w
    intrinsics[1, :] *= scale_h

    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

def read_semantic_mask(filename):
    s_mask = Image.open(filename)
    np_s_mask = np.array(s_mask, dtype=np.float32)
    return np_s_mask

def read_semantic_pfm(filename, save_cls):
    sem = read_pfm(filename)[0]
    mask = np.isin(sem, save_cls)
    return np.array(mask, dtype=np.float32)

def save_mask(filename, mask):
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = f.readline().rstrip()
            src_views = [x for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) == 0:
                continue
            data.append((ref_view, src_views))
    return data


def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                         depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]

    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks=[]
    for i in range(2,11):
        mask = np.logical_and(dist < i/4, relative_depth_diff < i/1300)
        masks.append(mask)
    # vis_mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected


def mp(params):

    scan_folder, img_folder, out_folder, pair_data, photo_threshold, p_idx = params
    
    for ref_view, src_views in tqdm(pair_data, desc=f"processID-{p_idx} fuse depth for {scan_folder.split('/')[-1]}"):
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(scan_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        

        # if True:
        ori_h, ori_w = confidence.shape[:2]
        confidence = cv2.resize(confidence, (int(ori_w * args.scale), int(ori_h * args.scale)))
        ref_depth_est = cv2.resize(ref_depth_est, (int(ori_w * args.scale), int(ori_h * args.scale)))

        img_ori_h, img_ori_w = ref_img.shape[:2]
        ref_img = cv2.resize(ref_img,(int(confidence.shape[1]),int(confidence.shape[0])))

        # read sem color map
        if args.save_sem_color:
            ref_sem_color = read_img(os.path.join(args.Semantic_path, scan_folder.split('/')[-1],
                                        "color_seg/{:0>8}.jpg".format(ref_view)))
            ref_sem_color = cv2.resize(ref_sem_color,(int(confidence.shape[1]),int(confidence.shape[0])))

        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)),
            scale_w=confidence.shape[1]/img_ori_w, scale_h=confidence.shape[0]/img_ori_h)

        # process semantic mask
        semantic_mask = read_semantic_pfm(os.path.join(args.Semantic_path, scan_folder.split('/')[-1],
                                        "seg/{:0>8}.pfm".format(ref_view)),
                                        save_cls=SAVE_CLS[args.Semantic_type])
        semantic_mask = cv2.resize(semantic_mask, (int(ref_img.shape[1]), int(ref_img.shape[0])))
        sem_mask = semantic_mask > 0.5
        # semantic_mask = (~(semantic_mask>0.)).astype(np.float32)
        photo_mask = confidence > photo_threshold
        all_srcview_depth_ests = []

        geo_mask_sum = 0
        geo_mask_sums=[]
        n=10
        ct = 0
        for src_view in src_views:
            ct = ct + 1
            src_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            # if args.scale != 1.0:
            ori_h, ori_w = src_depth_est.shape[:2]
            src_depth_est = cv2.resize(src_depth_est, (int(ori_w * args.scale), int(ori_h * args.scale)))

            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)),
                scale_w=confidence.shape[1]/img_ori_w, scale_h=confidence.shape[0]/img_ori_h)

            masks, geo_mask, depth_reprojected = check_geometric_consistency(ref_depth_est,
                                                                            ref_intrinsics,
                                                                            ref_extrinsics,
                                                                            src_depth_est,
                                                                            src_intrinsics,
                                                                            src_extrinsics)
            if (ct==1):
                for i in range(2,n):
                    geo_mask_sums.append(masks[i-2].astype(np.int32))
            else :
                for i in range(2,n):
                    geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)
            geo_mask_sum+=geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        # geo_mask=geo_mask_sum>=n
        geo_mask=geo_mask_sum>=args.thres_view
        for i in range (2,n):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        if (not isinstance(geo_mask, bool)):
            final_mask = np.logical_and(photo_mask, geo_mask)
            final_mask = np.logical_and(sem_mask, final_mask)
            
            if int(ref_view) % 25 == 0:
                save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
                save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
                save_mask(os.path.join(out_folder, "mask/{:0>8}_sem.png".format(ref_view)), sem_mask)
            if int(ref_view) % 5 == 0:
                save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            valid_points = final_mask
            # print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[:, :, :][valid_points]
            if args.save_sem_color:
                sem_color = ref_sem_color[:, :, :][valid_points]

            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]

        np.save(os.path.join(out_folder, "pts_npy/{:0>8}.npy".format(ref_view)), xyz_world.transpose(1, 0).reshape(-1,3))
        np.save(os.path.join(out_folder, "colors_npy/{:0>8}.npy".format(ref_view)), color.reshape(-1,3))
        if args.save_sem_color:
            np.save(os.path.join(out_folder, "sem_colors_npy/{:0>8}.npy".format(ref_view)), sem_color.reshape(-1,3))


def filter_depth_mp(scan_folder, img_folder, out_folder, pair_path, 
                    photo_threshold, p_num, save_sem_color=False):
    '''
    多线程对当前scan进行处理, 是在单个scan内进行截断加速
    '''
    # pair_file = pair_path
    pair_data = read_pair_file(pair_path)

    os.makedirs(os.path.join(out_folder, "pts_npy"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "colors_npy"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
    if save_sem_color:
        os.makedirs(os.path.join(out_folder, "sem_colors_npy"), exist_ok=True)

    params_mp = []
    pair_inter = math.ceil(len(pair_data) / p_num)
    for p_idx in range(p_num):
        id_s, id_e = p_idx*pair_inter, (p_idx+1)*pair_inter
        id_e = len(pair_data) if id_e >= len(pair_data) else id_e
        params_mp.append([scan_folder, img_folder, out_folder, pair_data[id_s:id_e], photo_threshold, p_idx])

    partial_func = partial(mp)
    p = Pool(p_num)
    p.map(partial_func, params_mp)
    p.close()
    p.join()


def main_v1(scans, p_num=8):
    for scan in scans:
        print(f"run depth fusion for {scan} with {p_num} process")

        scan_folder = os.path.join(args.testpath, scan)
        out_folder = os.path.join(args.outdir, scan)
        pair_path = os.path.join(args.input_path, f"{scan}/pair.txt")
        img_folder = os.path.join(args.input_path, scan)
        filter_depth_mp(scan_folder, img_folder, out_folder, pair_path, args.photo_threshold, p_num)


def mp_v2(params):

    scan_folder, img_folder, out_folder, pair_data, p_idx, \
        scale, save_sem_color, Semantic_path, photo_threshold, \
            thres_view, save_vis = params
    
    for ref_view, src_views in tqdm(pair_data, desc=f"processID-{p_idx} fuse depth for {scan_folder.split('/')[-1]}"):
        ref_img = read_img(os.path.join(img_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(scan_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        

        # if scale != 1.0:
        ori_h, ori_w = confidence.shape[:2]
        confidence = cv2.resize(confidence, (int(ori_w * scale), int(ori_h * scale)))
        ref_depth_est = cv2.resize(ref_depth_est, (int(ori_w * scale), int(ori_h * scale)))

        img_ori_h, img_ori_w = ref_img.shape[:2]
        ref_img = cv2.resize(ref_img,(int(confidence.shape[1]),int(confidence.shape[0])))

        # read sem color map
        if save_sem_color:
            ref_sem_color = read_img(os.path.join(Semantic_path, scan_folder.split('/')[-1],
                                        "color_seg/{:0>8}.jpg".format(ref_view)))
            ref_sem_color = cv2.resize(ref_sem_color,(int(confidence.shape[1]),int(confidence.shape[0])))

        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)),
            scale_w=confidence.shape[1]/ori_w, scale_h=confidence.shape[0]/ori_h)

        # process semantic mask

        semantic_mask = read_semantic_pfm(os.path.join(Semantic_path, scan_folder.split('/')[-1],
                                        "seg/{:0>8}.pfm".format(ref_view)),
                                        save_cls=SAVE_CLS['Cityscapes'])
        semantic_mask = cv2.resize(semantic_mask, (int(ref_img.shape[1]), int(ref_img.shape[0])))
        sem_mask = semantic_mask > 0.5
        
        # semantic_mask = (~(semantic_mask>0.)).astype(np.float32)
        photo_mask = confidence > photo_threshold
        all_srcview_depth_ests = []

        geo_mask_sum = 0
        geo_mask_sums=[]
        n=10
        ct = 0
        for src_view in src_views:
            #ct = ct + 1
            src_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            # if scale != 1.0:
            ori_h, ori_w = src_depth_est.shape[:2]
            src_depth_est = cv2.resize(src_depth_est, (int(ori_w * scale), int(ori_h * scale)))

            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)),
                scale_w=confidence.shape[1]/ori_w, scale_h=confidence.shape[0]/ori_h)
            
            # ext_dist = cal_ext_dist(ref_extrinsics,src_extrinsics) #mod
            # # print(ext_dist)
            # if ext_dist < 0.2:
            #     continue

            masks, geo_mask, depth_reprojected = check_geometric_consistency(ref_depth_est,
                                                                            ref_intrinsics,
                                                                            ref_extrinsics,
                                                                            src_depth_est,
                                                                            src_intrinsics,
                                                                            src_extrinsics)
            ct = ct + 1
            if (ct==1):
                for i in range(2,n):
                    geo_mask_sums.append(masks[i-2].astype(np.int32))
            else :
                for i in range(2,n):
                    geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)
            geo_mask_sum+=geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        # geo_mask=geo_mask_sum>=n
        geo_mask=geo_mask_sum>=thres_view
        for i in range (2,n):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        if (not isinstance(geo_mask, bool)):
            final_mask = np.logical_and(photo_mask, geo_mask)
            final_mask = np.logical_and(sem_mask, final_mask)
            
            if save_vis and int(ref_view) % 50 == 0:
                save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
                save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
                save_mask(os.path.join(out_folder, "mask/{:0>8}_sem.png".format(ref_view)), sem_mask)
            if save_vis and int(ref_view) % 20 == 0:
                save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            valid_points = final_mask
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[:, :, :][valid_points]
            if save_sem_color:
                sem_color = ref_sem_color[:, :, :][valid_points]

            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]

        np.save(os.path.join(out_folder, "pts_npy/{:0>8}.npy".format(ref_view)), xyz_world.transpose(1, 0).reshape(-1,3))
        np.save(os.path.join(out_folder, "colors_npy/{:0>8}.npy".format(ref_view)), color.reshape(-1,3))
        if save_sem_color:
            np.save(os.path.join(out_folder, "sem_colors_npy/{:0>8}.npy".format(ref_view)), sem_color.reshape(-1,3))


def filter_main_mp(scan, cfg):
    '''
    多线程对当前scan进行处理, 是在单个scan内进行截断加速
    '''
    p_num = cfg['mp_num']
    scan_folder = os.path.join(cfg['outdir'], scan)
    out_folder = os.path.join(cfg['outdir'], scan)
    pair_path = os.path.join(cfg['input_path'], f'{scan}/pair.txt')
    img_folder = os.path.join(cfg['input_path'], scan)
    pair_data = read_pair_file(pair_path)

    os.makedirs(os.path.join(out_folder, "pts_npy"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "colors_npy"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
    if cfg['save_sem_color']:
        os.makedirs(os.path.join(out_folder, "sem_colors_npy"), exist_ok=True)

    params_mp = []
    pair_inter = math.ceil(len(pair_data) / p_num)
    for p_idx in range(p_num):
        id_s, id_e = p_idx*pair_inter, (p_idx+1)*pair_inter
        id_e = len(pair_data) if id_e >= len(pair_data) else id_e
        params_mp.append([scan_folder, img_folder, out_folder, \
                        pair_data[id_s:id_e], p_idx, cfg['scale'], \
                        cfg['save_sem_color'], cfg['Semantic_path'], \
                        cfg['photo_threshold'], cfg['thres_view'], cfg['save_vis']])

    partial_func = partial(mp_v2)
    p = Pool(p_num)
    p.map(partial_func, params_mp)
    p.close()
    p.join()


def main(cfg):

    with open(cfg['testlist']) as f:
        scans = f.readlines()
        scans = sorted([line.rstrip() for line in scans])

    for scan in scans:
        print(f"run depth fusion for {scan} with {cfg['mp_num']} process")
        filter_main_mp(scan, cfg)


if __name__ == '__main__':

    with open(args.testlist) as f:
        scans = f.readlines()
        scans = sorted([line.rstrip() for line in scans])

    main_v1(scans, p_num=args.mp_num)
