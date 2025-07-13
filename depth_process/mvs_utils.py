import numpy as np
import open3d as o3d
import pickle
from PIL import Image
import cv2
import os
from pyquaternion import Quaternion
from tqdm import tqdm
import copy

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
        depth_trunc=100,  # 最大深度裁剪值
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
    transformation = np.linalg.inv(extrinsics)  # 如果外参为从世界到摄像头坐标系，需要取逆
    pcd.transform(transformation)

    return pcd


def merge_point_clouds_with_open3d(rgb_images, depth_images, intrinsics_list, extrinsics_list):
    """
    合并多个摄像头的点云。
    
    参数：
    - rgb_images: 所有摄像头的 RGB 图像列表
    - depth_images: 所有摄像头的深度图像列表
    - intrinsics_list: 所有摄像头的内参矩阵列表
    - extrinsics_list: 所有摄像头的外参矩阵列表
    
    返回：
    - merged_pcd: 合并后的 open3d.geometry.PointCloud 对象
    """
    merged_pcd = o3d.geometry.PointCloud()

    for rgb, depth, intrinsics, extrinsics in zip(rgb_images, depth_images, intrinsics_list, extrinsics_list):
        pcd = create_point_cloud_with_open3d(rgb, depth, intrinsics, extrinsics)
        merged_pcd += pcd

    # 对合并后的点云进行降采样（可选，减少点云大小）
    # merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)
    return merged_pcd

def resize_max_res_cv2(img: np.ndarray, max_edge_resolution: int, interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`np.ndarray`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `np.ndarray`: Resized image.
    """
    
    original_height, original_width = img.shape[:2]
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return resized_img

def to_matrix4x4_2(rotation, translation, inverse=True):
    output = np.eye(4)
    output[:3, :3] = rotation
    output[:3, 3]  = translation
    
    if inverse:
        output = np.linalg.inv(output)
    return output

def save_MVS_cam_txt(cam_dir,i,intrinsic,extrinsic,depth_ranges):
    with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
        f.write('extrinsic\n')
        for j in range(4):
            for k in range(4):
                f.write(str(extrinsic[j, k]) + ' ')
            f.write('\n')
        f.write('\nintrinsic\n')
        for j in range(3):
            for k in range(3):
                f.write(str(intrinsic[j, k]) + ' ')
            f.write('\n')
        f.write('\n%f %f %f %f\n' % (depth_ranges[0], depth_ranges[1], depth_ranges[2], depth_ranges[3]))


def read_MVS_pcd(sp_pcd_root):
    pcd_list = os.listdir(sp_pcd_root)
    merged_pcd = o3d.geometry.PointCloud()
    for pcd_path in pcd_list:
        pcd = o3d.io.read_point_cloud(os.path.join(sp_pcd_root,pcd_path))
        merged_pcd+=pcd
    return merged_pcd

def pair_from_front_id(img_path, save_file):
    '''根据id顺序, 优先考虑前向的id'''

    imgs = sorted(os.listdir(img_path))

    length = len(imgs)
    with open(save_file, 'w') as f:
        f.writelines(f"{length}\n")

        for i in tqdm(range(length), desc="generate pair.txt"):
            img = imgs[i]
            f.writelines(f"{int(img.split('.')[0])}\n")
            f.writelines('10 ')

            if i < 5:
                count = 0
                for j in range(i):
                    f.writelines(f"{int(imgs[i-1-j].split('.')[0])} {100.-j} {int(imgs[i+1+j].split('.')[0])} {100.-j-0.2} ")
                    count += 2
                for j in range(10-count):
                    f.writelines(f"{int(imgs[i+1+int(count/2)+j].split('.')[0])} {50-j} ")

            elif i >= length - 5 - 1:
                count = 0
                for j in range(length-1-i):
                    f.writelines(f"{int(imgs[i-1-j].split('.')[0])} {100.-j} {int(imgs[i+1+j].split('.')[0])} {100.-j-0.2} ")
                    count += 2
                for j in range(10-count):
                    f.writelines(f"{int(imgs[i-1-j-int(count/2)].split('.')[0])} {50-j} ")

            else:# 优先考虑前向的frame id
                if i >= length - 10 -1:
                    for j in range(5):
                        f.writelines(f"{int(imgs[i-1-j].split('.')[0])} {100.-j} {int(imgs[i+1+j].split('.')[0])} {100.-j-0.2} ")
                else:#  i => [5, N-10]
                    for item in range(3):
                        back_item = item+1    # 1,2,3
                        front_item_1, front_item_2 = item*2+1, item*2+2     # (1,2) (3,4) (5,6)
                        f.writelines(f"{int(imgs[i+front_item_1].split('.')[0])} {100-front_item_1} ")
                        f.writelines(f"{int(imgs[i+front_item_2].split('.')[0])} {100-front_item_2} ")
                        f.writelines(f"{int(imgs[i-back_item].split('.')[0])} {100-back_item*2} ")
                    f.writelines(f"{int(imgs[i-4].split('.')[0])} {100-20} ")
            f.writelines('\n')



