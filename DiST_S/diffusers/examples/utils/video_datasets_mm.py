import os, re
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as tr
import random, json
import pickle as pkl
import tqdm
import cv2
import random
from PIL import Image, ImageDraw
from collections import defaultdict
from utils.image_datasets import combine_text_conds, get_text_cond_tokens, TEXT_CONDITIONS, default_loader
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
# from .image_datasets import default_loader
from ipdb import set_trace
import matplotlib


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


class Video_reproj_nus_multi_model_Dataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        split_name,
        pesudo_flag_name_list,
        video_transforms=None,
        single_cam=False,
        with_valid_mask=False,
        pros_w=512,
        pros_h=320,
        **kargs
        ):

        super().__init__()
        # self.mismatch_aug_ratio=mismatch_aug_ratio
        self.pros_w = pros_w
        self.pros_h = pros_h
        self.loader = default_loader
        # self.init_caption = init_caption 
        # self.multi_view = multi_view
        self.video_length = video_length
        self.img_transform = video_transforms

        rain_list = json.load(open("../../misc/rain_night_scene_list.json", 'r'))
        self.rain_list = rain_list[split_name]

        self.with_valid_mask = with_valid_mask

        if single_cam:
            self.camera_view = ['CAM_FRONT']
            print(f"===========! single_cam: {single_cam} !===========")
        else:
            self.camera_view = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
        
        self.videos,self.pesudo_video_clips, self.depth_videos, self.pesudo_depth_videos = self._make_dataset_video(data_root,
                                                                                                                    split_name,
                                                                                                                    pesudo_flag_name_list)
        # if self.ego:
        #     self.videos, self.conds = self._make_dataset_video_ego(data_root, video_length)
        # else:
        #     self.videos, self.conds = self._make_dataset_video(data_root, video_length)

        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )
        # import pdb;pdb.set_trace()
        # self.intrin_aug_range=[0.5, 1]

    def __len__(self):
        return len(self.videos)
    
    def _make_dataset_video(self,data_root,split_name,pesudo_flag_name_list):

        video_clips = []
        pesudo_video_clips = []
        depth_video_clips = []
        pesudo_depth_video_clips = []
        
        for pesudo_flag_name in pesudo_flag_name_list:
            
            split_root = f"{data_root}/{split_name}/{pesudo_flag_name}"
            if pesudo_flag_name=="cycle_reproj":
                split_root="../reproj_nus/cycle_reproj/train"
            scene_list = os.listdir(split_root)
            scene_list = [scene_ for scene_ in scene_list if int(scene_) not in self.rain_list]

            # allscene_sample_token_list=[]
            
            for scene in scene_list:
                with open(f"{split_root}/{scene}/pesudo_sampletoken.json",'r') as file:
                    sample_token_list = json.load(file) 

                if pesudo_flag_name=="cycle_reproj":
                    sample_token_list = sample_token_list[:-5]
                scene_sample_len = len(sample_token_list)
                for cam in self.camera_view:
                    scene_video_clipes = []
                    scene_pesudo_video_clips = []
                    scene_depth_video_clips = []
                    scene_pesudo_depth_video_clips = []
                    for idx in range(scene_sample_len):
                        pesudo_img_filepath = f"{split_root}/{scene}/{cam}/{sample_token_list[idx]}.jpg"
                        img_filepath = f"{data_root}/{split_name}/frame_gt/{scene}/{cam}/{sample_token_list[idx]}.jpg"
                        depth_img_filepath = f"{data_root}/{split_name}/frame_gt/{scene}/{cam}/{sample_token_list[idx]}.npz"
                        pesudo_depth_img_filepath = f"{split_root}/{scene}/{cam}/{sample_token_list[idx]}.npz"
                        scene_video_clipes.append(img_filepath)
                        scene_pesudo_video_clips.append(pesudo_img_filepath)
                        scene_depth_video_clips.append(depth_img_filepath)
                        scene_pesudo_depth_video_clips.append(pesudo_depth_img_filepath)
                    video_clips.append(scene_video_clipes)
                    pesudo_video_clips.append(scene_pesudo_video_clips)
                    depth_video_clips.append(scene_depth_video_clips)
                    pesudo_depth_video_clips.append(scene_pesudo_depth_video_clips)

        return video_clips, pesudo_video_clips, depth_video_clips, pesudo_depth_video_clips
    
    def aug_mask_peudoimage(self, pesudo_image, random_p=0.2, random_aug_ratio=0.35, mask_aug_shape_ratio=0.3):
        
        np_img = np.array(pesudo_image)
        ori_mask = (np.sum(np_img, axis=-1) == 0).astype(np.uint8)*255

        # debug_path = f"/data/gen_work/code/FreeVS/diffusers/work_dirs/debug/"
        # Image.fromarray(np_img).save(f"{debug_path}/debug_original.jpg")
        
        # random mask pixels
        new_mask_random = np.zeros_like(ori_mask)[:,:]
        random_var = np.random.rand()
        if random_var < random_p:
            rand = np.random.rand(*new_mask_random.shape)
            new_mask_random[rand < random_aug_ratio] = 1

        # mask shape
        predefined_shapes = [
            [(10, 10), (50, 10), (40, 50), (20, 40)],
            [(15, 15), (45, 15), (45, 45), (15, 45)],
            [(30, 5), (55, 30), (30, 55), (5, 30)],
            [(20, 10), (40, 10), (50, 30), (40, 50), (20, 50), (10, 30)],
            [(25, 5), (35, 5), (45, 25), (35, 45), (25, 45), (15, 25)],
            [(10, 20), (30, 10), (50, 20), (40, 40), (20, 40)],
            [(5, 30), (25, 10), (45, 30), (35, 50), (15, 50)],
            [(20, 20), (40, 20), (40, 40), (20, 40)],
        ]
        large_shapes = [
            [(25, 15), (35, 15), (45, 30), (35, 45), (25, 45), (15, 30)],
            [(10, 25), (30, 10), (50, 25), (40, 40), (20, 40)],
            [(10, 10), (100, 10), (90, 90), (20, 80)],
            [(20, 20), (90, 20), (90, 80), (20, 80)],
            [(30, 5), (80, 30), (30, 95), (5, 30)],
            [(15, 15), (95, 15), (95, 85), (15, 85)],
            [(25, 10), (85, 10), (95, 50), (85, 90), (25, 90), (10, 50)],
            [(10, 20), (50, 10), (100, 20), (80, 80), (20, 80)],
            [(5, 30), (50, 10), (105, 30), (85, 70), (15, 70)],
            [(20, 20), (90, 20), (90, 80), (20, 80)],
            [(25, 15), (85, 15), (95, 50), (85, 85), (25, 85), (15, 50)],
            [(10, 25), (50, 10), (100, 25), (80, 75), (20, 75)]]
        
        mask_shape = np.zeros_like(ori_mask)[:,:]
        random_var = np.random.rand()
        if random_var < mask_aug_shape_ratio:
            
            mask_image = Image.fromarray((mask_shape * 255).astype(np.uint8))
            draw = ImageDraw.Draw(mask_image)
            width, height = mask_image.size

            num_shapes_to_add = random.randint(1, 5)
            selected_shapes = random.sample(predefined_shapes, num_shapes_to_add)
            num_large_shapes_to_add = random.randint(1, 3)
            selected_shapes += random.sample(large_shapes, num_large_shapes_to_add)

            for shape in selected_shapes:
                max_x = width - 110
                max_y = height - 110
                top_left_x = random.randint(0, max_x)
                top_left_y = random.randint(0, max_y)    
                translated_shape = [(x + top_left_x, y + top_left_y) for (x, y) in shape]    
                draw.polygon(translated_shape, fill=1)

            mask_shape = np.array(mask_image) #// 255

        np_img[new_mask_random == 1, :] = 0
        np_img[mask_shape == 1, :] = 0
        
        # if not os.path.exists(debug_path):
        #     os.makedirs(debug_path)
        # Image.fromarray(np_img).save(f"{debug_path}/debug_mask_aug.jpg")
        # exit()

        return Image.fromarray(np_img), new_mask_random, mask_shape

    def load_depth(self, file_name, depth_min=0.5, depth_max=100.0):
        # import pdb; pdb.set_trace()
        if "gt" in file_name:
            depth = np.load(file_name)['depth_gt']
        else:
            depth = np.load(file_name)['depth_proj']
        depth = np.clip(depth, depth_min, depth_max)
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        depth /= depth_max

        return depth
    
    def aug_mask_peudoimage_depth(self, pesudo_depth_img, mask_random, mask_shape,depth_min=0.5, depth_max=100.0):
        pad_value = depth_min / depth_max
        np_img = np.array(pesudo_depth_img)
        np_img[mask_random == 1, :] = pad_value
        np_img[mask_shape == 1, :] = pad_value

        return np_img

    def load_and_transform_frames(self, frame_list, pseudo_frame_list, depth_frame_list, pesudo_depth_frame_list,
                                  loader, img_transform=None, depth_min=0.5, depth_max=100.0):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []
        pseudo_clip=[]
        depth_clip = []
        pesudo_depth_clip = []

        # import pdb; pdb.set_trace()
        for frame, pseudo_frame, depth_frame, pesudo_depth_frame in zip(frame_list,pseudo_frame_list,depth_frame_list,pesudo_depth_frame_list):
            img = loader(frame)
            pseudo_img = loader(pseudo_frame)
            depth_img = self.load_depth(depth_frame, depth_min=depth_min, depth_max=depth_max)
            pesudo_depth_img = self.load_depth(pesudo_depth_frame, depth_min=depth_min, depth_max=depth_max)

            # add mask aug
            # pseudo_img, new_mask_random, mask_shape = self.aug_mask_peudoimage(pseudo_img,
            #                                                                    random_p=0.2,
            #                                                                     random_aug_ratio=0.25,
            #                                                                     mask_aug_shape_ratio=0.2)
            
            # add mask aug for depth
            # pesudo_depth_img = self.aug_mask_peudoimage_depth(pesudo_depth_img, new_mask_random, mask_shape)
            # depth_img = Image.fromarray(depth_img)

            # pesudo_depth_img = self.aug_mask_peudoimage_depth(pesudo_depth_img, new_mask_random, mask_shape,
            #                                                   depth_min=depth_min, depth_max=depth_max)

            if self.with_valid_mask:
                valid_mask = (pesudo_depth_img[:,:,-1] > depth_min/depth_max).astype(np.float32)
                pesudo_depth_img[:,:,-1] = valid_mask


            img =img.resize((self.pros_w,self.pros_h))
            pseudo_img =pseudo_img.resize((self.pros_w,self.pros_h))

            # depth_img = depth_img.resize((512,320), resample=Image.Resampling.NEAREST)
            # pesudo_depth_img = pesudo_depth_img.resize((512,320), resample=Image.Resampling.NEAREST)
            depth_img = cv2.resize(depth_img, (self.pros_w,self.pros_h), interpolation=cv2.INTER_NEAREST)
            pesudo_depth_img = cv2.resize(pesudo_depth_img, (self.pros_w,self.pros_h), interpolation=cv2.INTER_NEAREST)

            if img_transform is not None:
                img = img_transform(img)
                pseudo_img = img_transform(pseudo_img)
                depth_img = img_transform(depth_img)
                pesudo_depth_img = img_transform(pesudo_depth_img)
            else:
                # raise NotImplementedError("must give img_transform")
                img = self.default_transforms(img)
                pseudo_img = self.default_transforms(pseudo_img)
                depth_img = self.default_transforms(depth_img)
                pesudo_depth_img = self.default_transforms(pesudo_depth_img)
            
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            pseudo_img = pseudo_img.view(pseudo_img.size(0),1, pseudo_img.size(1), pseudo_img.size(2))
            depth_img = depth_img.view(depth_img.size(0),1, depth_img.size(1), depth_img.size(2))
            pesudo_depth_img = pesudo_depth_img.view(pesudo_depth_img.size(0),1, pesudo_depth_img.size(1), pesudo_depth_img.size(2))

            clip.append(img)
            pseudo_clip.append(pseudo_img)
            depth_clip.append(depth_img)
            pesudo_depth_clip.append(pesudo_depth_img)

        return clip,  pseudo_clip, depth_clip, pesudo_depth_clip
    
        

    def __getitem__(self, index):
        video, conds = self.videos[index], self.pesudo_video_clips[index]
        depth_video, pesudo_depth_video = self.depth_videos[index], self.pesudo_depth_videos[index]
        
        init_frame = random.randint(0,len(video)-self.video_length)
        video = video[init_frame:init_frame+self.video_length] # video begin with a random frame
        conds = conds[init_frame:init_frame+self.video_length]
        depth_video = depth_video[init_frame:init_frame+self.video_length]
        pesudo_depth_video = pesudo_depth_video[init_frame:init_frame+self.video_length]

        assert(len(video) == self.video_length)

        frames, pseudo_frames, depth_frames, pesudo_depth_frames = self.load_and_transform_frames(video, conds, 
                                                                            depth_video, pesudo_depth_video,
                                                                            self.loader, self.img_transform)

        frames = torch.cat(frames, 1) # c,t,h,w
        frames = frames.transpose(0, 1) # t,c,h,w

        pseudo_frames = torch.cat(pseudo_frames, 1) # c,t,h,w
        pseudo_frames = pseudo_frames.transpose(0, 1) # t,c,h,w

        depth_frames = torch.cat(depth_frames, 1) # c,t,h,w
        depth_frames = depth_frames.transpose(0, 1) # t,c,h,w

        pesudo_depth_frames = torch.cat(pesudo_depth_frames, 1) # c,t,h,w
        pesudo_depth_frames = pesudo_depth_frames.transpose(0, 1) # t,c,h,w

        example = dict()
        example["pixel_values"] = frames
        example["pseudo_pixel_values"] = pseudo_frames
        example["depth_values"] = depth_frames
        example["pseudo_depth_values"] = pesudo_depth_frames
        example["images"],_,example['depth'],_ = self.load_and_transform_frames(video, conds, depth_video, pesudo_depth_video, self.loader)

        
        return example


if __name__ == "__main__":

    from torchvision import transforms

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = Video_reproj_nus_multi_model_Dataset(
        data_root="./reproj_nus",
        video_length=6,
        split_name="train",
        pesudo_flag_name="frame_pesudo+2",
        video_transforms=train_transforms,
    )

    item = dataset.__getitem__(0)
    import pdb; pdb.set_trace()
    print(item.keys())