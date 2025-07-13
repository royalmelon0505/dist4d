import os, time, sys, gc, cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from MVS.TransMVSNet_hr.utils import *
from MVS.TransMVSNet_hr.datasets import find_dataset_def
from MVS.TransMVSNet_hr.models.TransMVSNet import TransMVSNet
from MVS.TransMVSNet_hr.datasets.data_io import read_pfm, save_pfm
from PIL import Image


cudnn.benchmark = True

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def save_scene_depth(testlist, cfg):
    # dataset, dataloader
    MVSDataset = find_dataset_def(cfg['dataset'])
    test_dataset = MVSDataset(cfg['testpath'], testlist, "test", cfg['num_view'], cfg['numdepth'], cfg['interval_scale'],
                              max_h=cfg['max_h'], max_w=cfg['max_w'], fix_res=cfg['fix_res'])
    TestImgLoader = DataLoader(test_dataset, cfg['batch_size'], shuffle=False, num_workers=cfg['num_worker'], drop_last=False)

    # model
    num_stage = len(cfg['ndepths'])
    model = TransMVSNet(refine=False, ndepths=cfg['ndepths'],
                        depth_interals_ratio=cfg['depth_inter_r'],
                        cr_base_chs=cfg['cr_base_chs'])


    print("loading model {}".format(cfg['loadckpt']))
    state_dict = torch.load(cfg['loadckpt'], map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in tqdm(enumerate(TestImgLoader), total=len(TestImgLoader), desc='mvs depth inference', colour='GREEN'):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()
            # print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, photometric_confidence, conf_1, conf_2 in zip(filenames, cams, imgs, \
                                                            outputs["depth"], outputs["photometric_confidence"],  outputs['stage1']["photometric_confidence"], outputs['stage2']["photometric_confidence"]):
                img = img[0]  #ref view
                cam = cam[0]  #ref cam
                H,W = photometric_confidence.shape
                conf_1 = cv2.resize(conf_1, (W,H))
                conf_2 = cv2.resize(conf_2, (W,H))
                conf_final = photometric_confidence * conf_1 * conf_2

                depth_filename = os.path.join(cfg['outdir'], filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(cfg['outdir'], filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(cfg['outdir'], filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(cfg['outdir'], filename.format('images', '.jpg'))
                # ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)

                save_pfm(depth_filename, depth_est)
                if cfg['save_vis'] and batch_idx % 1 == 0:
                    depth_color = visualize_depth(depth_est)
                    cv2.imwrite(os.path.join(cfg['outdir'], filename.format('depth_est', '.jpg')), depth_color)
                
                #save confidence maps
                save_pfm(confidence_filename, conf_final)
                write_cam(cam_filename, cam)
                if cfg['save_rgb']:
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)


    torch.cuda.empty_cache()
    gc.collect()


def main(cfg):
    
    with open(cfg['testlist']) as f:
        content = f.readlines()
        testlist = [line.rstrip() for line in content]
    
    for scene in testlist:
        save_scene_depth([scene], cfg)

