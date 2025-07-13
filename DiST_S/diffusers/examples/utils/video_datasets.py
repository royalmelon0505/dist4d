import os, re
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as tr
import random, json
import pickle as pkl
import tqdm
from collections import defaultdict
from utils.image_datasets import combine_text_conds, get_text_cond_tokens, TEXT_CONDITIONS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .image_datasets import default_loader
from ipdb import set_trace


class VideoNuscenesDataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        video_transforms=None,
        text_tokenizer=None,
        init_caption='',
        multi_view=False,
        ego = False,
        mismatch_aug_ratio = None,
        **kargs
        ):

        super().__init__()
        self.mismatch_aug_ratio=mismatch_aug_ratio
        self.loader = default_loader
        self.init_caption = init_caption 
        self.multi_view = multi_view
        self.video_length = video_length
        self.img_transform = video_transforms
        # self.camera_view = ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']
        # TODO: hardcode here
        self.camera_view = ['CAM_FRONT']
        # self.camera_captions = {'cam01': 'front camera view',
        #                         'CAM_BACK': 'back camera view',
        #                         'CAM_FRONT': 'front camera view',
        #                         'CAM_FRONT_RIGHT':'front right camera view',
        #                         'CAM_FRONT_LEFT':'front left camera view',
        #                         'CAM_BACK_RIGHT':'back right camera view',
        #                         'CAM_BACK_LEFT':'back left camera view'
        #                         }

        self.ego = ego

        if self.ego:
            self.videos, self.conds = self._make_dataset_video_ego(data_root, video_length)
        else:
            self.videos, self.conds = self._make_dataset_video(data_root, video_length)

        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )
        # import pdb;pdb.set_trace()
        self.intrin_aug_range=[0.5, 1]

    def __getitem__(self, index):
        video, conds = self.videos[index], self.conds[index]
        init_frame = random.randint(0,len(video)-self.video_length)
        video = video[init_frame:init_frame+self.video_length] # video begin with a random frame
        conds = conds[init_frame:init_frame+self.video_length]
        assert(len(video) == self.video_length)

        
        # make clip tensor
        if self.multi_view:
            print('not support yet')
        else:
            frames,pseudo_frames = self.load_and_transform_frames(video, self.loader, self.img_transform)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w

            pseudo_frames = torch.cat(pseudo_frames, 1) # c,t,h,w
            pseudo_frames = pseudo_frames.transpose(0, 1) # t,c,h,w
            
            example = dict()
            example["pixel_values"] = frames
            example["pseudo_pixel_values"] = pseudo_frames
            example["images"],_ = self.load_and_transform_frames(video, self.loader)
            if self.ego:
                frames_ego = torch.tensor(conds)
                example["ego_values"] = frames_ego / (0.5)
        
        return example

    def __len__(self):
        return len(self.videos)
    
    def _make_dataset_video_ego(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)

        
        output_videos = []
        output_ego_videos = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            for cam_v in self.camera_view:
                view_video = []
                ego_video = []
                for frame_i in range(len(frames[cam_v])):
                    frame = frames[cam_v][frame_i]
                    frame_path = frame
                    view_video.append(frame_path)
                    ego_video.append(frames['ego'][frame_i])
                
                output_videos.append(view_video)
                output_ego_videos.append(ego_video)

        return output_videos, output_ego_videos

    def _make_dataset_video(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        # print(video_info.keys())

        output_videos = []
        output_text_videos = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            # import pdb;pdb.set_trace()
            if not frames[0]['image'].endswith("FRONT.jpg"):
                continue
            for cam_v in self.camera_view:
                view_video = []
                text_video = []
                # import pdb;pdb.set_trace()
                for frame in frames:
                    view_video.append(frame)
                    # text_video.append(self.camera_captions[cam_v])
                    text_video.append(frame)
                
                output_videos.append(view_video)
                output_text_videos.append(text_video)

            
        # 

        return output_videos, output_text_videos
    
    def _make_dataset(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_text_videos = []
        if not self.multi_view:
            for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
                for cam_v in self.camera_view:
                    view_video = []
                    text_video = []
                    for frame in frames:
                        view_video.append(frame[cam_v])
                        text_video.append(self.camera_captions[cam_v])
                    
                    output_videos.append(view_video)
                    output_text_videos.append(text_video)
        else:
            for video_name, frames in video_info.items():
                output_videos.append(frames)

        return output_videos, output_text_videos

    def load_and_transform_frames(self, frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []
        pseudo_clip=[]

        use_mismatch_aug=False
        if np.random.rand()<self.mismatch_aug_ratio:
            use_mismatch_aug=True
            minusaug=False
            if np.random.rand()<0.5:
                minusaug=True

        use_intrin_aug=False
        intrin_aug_scale = np.random.rand()*(self.intrin_aug_range[1]-self.intrin_aug_range[0])+self.intrin_aug_range[0]

        # import pdb;pdb.set_trace()

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["image"]
                pseudo_fpath=frame["pseudo_image"]
                
                if use_mismatch_aug:
                    if minusaug:
                        pseudo_fpath=frame["pseudo_image_minusframe"]
                    else:
                        pseudo_fpath=frame["pseudo_image_plusframe"]
                # print(pseudo_fpath)
                pseudo_fpath = pseudo_fpath.replace("waymo_pseudoimg_multiframe_example","waymo_pseudoimg_multiframe_democases")
            else:
                fpath = frame
            
            # print(pseudo_fpath)

            #960,384
            img = loader(fpath)
            try:
                pseudo_img = loader(pseudo_fpath)
            except:
                pseudo_fpath=frame["pseudo_image"]
                pseudo_img = loader(pseudo_fpath)

            img_w, img_h = img.size[:2]
            if use_intrin_aug:#self.video_center_crop:
                # import pdb
                # pdb.set_trace()
                newh = int(img_h*intrin_aug_scale)
                neww = int(img_w*intrin_aug_scale)
                h_diff = (img_h - newh) // 2
                w_diff = (img_w - neww) // 2
                left = w_diff
                upper = h_diff
                right = w_diff + neww
                lower = h_diff + newh

                # Ensure the crop area is within the image bounds
                if right > img_w:
                    right = img_w
                if lower > img_h:
                    lower = img_h
                img = img.crop((left, upper, right, lower))
                pseudo_img = pseudo_img.crop((left, upper, right, lower))

                img = img.resize((img_w,img_h))
                pseudo_img = pseudo_img.resize((img_w,img_h))

            # #Free-VS on Waymo
            # None, or#pseudo_img = pseudo_img.resize((int(960),int(384)))

            if img_transform is not None:
                img = img_transform(img)
                pseudo_img = img_transform(pseudo_img)
            else:
                img = self.default_transforms(img)
                pseudo_img = self.default_transforms(pseudo_img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            pseudo_img = pseudo_img.view(pseudo_img.size(0),1, pseudo_img.size(1), pseudo_img.size(2))
            clip.append(img)
            pseudo_clip.append(pseudo_img)
        return clip,  pseudo_clip

class VideoCondNuscenesDataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        video_transforms=None,
        text_tokenizer=None,
        init_caption='',
        multi_view=False,
        conditions = None,
        **kargs
        ):

        super().__init__()
        self.loader = default_loader
        self.init_caption = init_caption 
        self.multi_view = multi_view
        self.video_length = video_length
        self.img_transform = video_transforms
        # self.camera_view = ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']
        self.camera_view = ['CAM_FRONT']
        self.camera_captions = {'CAM_BACK': 'back camera view',
                                'CAM_FRONT': 'front camera view',
                                'CAM_FRONT_RIGHT':'front right camera view',
                                'CAM_FRONT_LEFT':'front left camera view',
                                'CAM_BACK_RIGHT':'back right camera view',
                                'CAM_BACK_LEFT':'back left camera view'
                                }
        self.conditions = conditions

        self.videos, self.conds = self._make_dataset(data_root, video_length, conditions = conditions)

        self.cond_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )
        self.img_w = 384

    def __getitem__(self, index):

        video = self.videos[index]
        init_frame = random.randint(0,len(video)-self.video_length)
        video = video[init_frame:init_frame+self.video_length] # video begin with a random frame
        
        if 'box' in self.conditions:
            conds_box = self.conds['box'][index]
            conds_box = conds_box[init_frame:init_frame+self.video_length]
        if 'map' in self.conditions:
            conds_map = self.conds['map'][index]
            conds_map = conds_map[init_frame:init_frame+self.video_length]
        if 'bev' in self.conditions:
            conds_bev = self.conds['bev'][index]
            conds_bev = conds_bev[init_frame:init_frame+self.video_length]
        if 'text' in self.conditions:
            conds_text = self.conds['text'][index]
            conds_text = conds_text[init_frame:init_frame+self.video_length]

        assert(len(video) == self.video_length)

        # make clip tensor
        if self.multi_view:
            print('not support yet')
        else:
            frames = self.load_and_transform_frames(video, self.loader, self.img_transform)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w

            example = dict()

            if 'map' in self.conditions:
                frames_map = self.load_and_transform_frames_map(conds_map, self.loader, self.cond_transforms)
                frames_map = torch.cat(frames_map,1)
                frames_map = frames_map.transpose(0,1)
                example["map_values"] = frames_map
            if 'box' in self.conditions:
                frames_box = self.load_and_transform_frames_box(conds_box, self.loader, self.cond_transforms)
                frames_box = torch.cat(frames_box,1)
                frames_box = frames_box.transpose(0,1)
                example["box_values"] = frames_box
            if 'bev' in self.conditions:
                frames_bev = self.load_and_transform_frames_map(conds_bev, self.loader, self.cond_transforms)
                frames_bev = torch.cat(frames_bev,1)
                frames_bev = frames_bev.transpose(0,1)
                example["bev_values"] = frames_bev
            if 'text' in self.conditions:
                # not support yet
                example["text_values"] = conds_text
    
            example["pixel_values"] = frames
            example["images"] = self.load_and_transform_frames(video, self.loader)
        
        return example

    def __len__(self):
        return len(self.videos)
    
    def _make_dataset(self, info_path, nframes, conditions):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        use_map, use_box, use_bev, use_text = False, False, False, False
        if 'map' in conditions:
            output_map_videos = []
            use_map = True
        if 'box' in conditions:
            output_box_videos = []
            use_box = True
        if 'bev' in conditions:
            output_bev_videos = []
            use_bev = True
        if 'text' in conditions:
            output_text_videos = []
            use_text = True

        if not self.multi_view:
            for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
                for cam_v in self.camera_view:
                    view_video = []
                    text_video = []
                    map_video  = []
                    box_video  = []
                    bev_video = []
                    for frame in frames:
                        view_video.append(frame[cam_v])
                        if use_map:
                            map_name = cam_v+'_map_rgb'
                            map_video.append(frame[map_name])
                        if use_box:
                            box_name = cam_v + '_box'
                            box_video.append(frame[box_name])
                        if use_bev:
                            bev_name = cam_v + '_bev'
                            bev_video.append(frame[bev_name])
                        if use_text:
                            text_video.append(self.camera_captions[cam_v])
                    
                    output_videos.append(view_video)
                    if use_text:
                        output_text_videos.append(text_video)
                    if use_map:
                        output_map_videos.append(map_video)
                    if use_box:
                        output_box_videos.append(box_video)
                    if use_bev:
                        output_bev_videos.append(bev_video)
        else:
            for video_name, frames in video_info.items():
                output_videos.append(frames)
        
        output_cond_videos = {}
        if use_map:
            output_cond_videos['map'] = output_map_videos
        if use_box:
            output_cond_videos['box'] = output_box_videos
        if use_bev:
            output_cond_videos['bev'] = output_bev_videos
        if use_text:
            output_cond_videos['text'] = output_text_videos

        return output_videos, output_cond_videos

    def load_and_transform_frames(self, frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            img = loader(fpath)
            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

    def load_and_transform_frames_map(self, frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            
            img = loader(fpath)
            if img_transform is not None:
                img = img_transform(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

    def load_and_transform_frames_box(self, frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            
            img = loader(fpath)
            crop_img = img.crop([0,100,img.size[0],img.size[1]])
            crop_img = crop_img.resize((self.img_w,self.img_w//2))
            if img_transform is not None:
                img = img_transform(crop_img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

class Video_reproj_nus_Dataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        split_name,
        # pesudo_flag_name,
        pesudo_flag_name_list,
        video_transforms=None,
        **kargs
        ):

        super().__init__()
        # self.mismatch_aug_ratio=mismatch_aug_ratio
        self.loader = default_loader
        # self.init_caption = init_caption 
        # self.multi_view = multi_view
        self.video_length = video_length
        self.img_transform = video_transforms
        self.camera_view = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT' , 'CAM_BACK', 'CAM_BACK_LEFT']
        
        self.videos,self.pesudo_video_clips = self._make_dataset_video(data_root,split_name,pesudo_flag_name_list)
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
        for pesudo_flag_name in pesudo_flag_name_list:
            split_root = f"{data_root}/{split_name}/{pesudo_flag_name}"
            scene_list = os.listdir(split_root)
            # allscene_sample_token_list=[]
            # video_clips = []
            # pesudo_video_clips = []
            with open("/mnt/bringup/crosshairs/guojiazhe/reproj_nus/rain_night_scene_list.json",'r') as file:
                rain_night_scene_list = json.load(file)
            rain_night_scene_list_use = rain_night_scene_list[split_name]
            for scene in scene_list:
                if int(scene) in rain_night_scene_list_use:
                    print(f"skip scene:{scene}")
                    continue

                with open(f"{split_root}/{scene}/pesudo_sampletoken.json",'r') as file:
                    sample_token_list = json.load(file) 
                # allscene_sample_token_list.append(sample_token_list)
                scene_sample_len = len(sample_token_list)
                for cam in self.camera_view:
                    scene_video_clipes = []
                    scene_pesudo_video_clips = []
                    for idx in range(scene_sample_len):
                        pesudo_img_filepath = f"{split_root}/{scene}/{cam}/{sample_token_list[idx]}.jpg"
                        img_filepath = f"{data_root}/{split_name}/frame_gt/{scene}/{cam}/{sample_token_list[idx]}.jpg"
                        scene_video_clipes.append(img_filepath)
                        scene_pesudo_video_clips.append(pesudo_img_filepath)
                    video_clips.append(scene_video_clipes)
                    pesudo_video_clips.append(scene_pesudo_video_clips)
            # import pdb;pdb.set_trace()
        return video_clips,pesudo_video_clips

    def load_and_transform_frames(self, frame_list, pesudo_frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []
        pseudo_clip=[]

        for frame,pesudo_frame in zip(frame_list,pesudo_frame_list):
            img = loader(frame)
            pseudo_img = loader(pesudo_frame)

            img =img.resize((512,320))
            pseudo_img =pseudo_img.resize((512,320))
            

            # image_array = np.array(pseudo_img)
            # height, width, _ = image_array.shape
            # random_mask = np.random.rand(height, width)
            # mask = random_mask < 0.8
            # # print(sum(mask))
            # image_array[mask, :] = [0, 0, 0]
            # pseudo_img = Image.fromarray(image_array)


            if img_transform is not None:
                img = img_transform(img)
                pseudo_img = img_transform(pseudo_img)
            else:
                img = self.default_transforms(img)
                pseudo_img = self.default_transforms(pseudo_img)
            
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            pseudo_img = pseudo_img.view(pseudo_img.size(0),1, pseudo_img.size(1), pseudo_img.size(2))

            clip.append(img)
            pseudo_clip.append(pseudo_img)
        return clip,  pseudo_clip
    
        

    def __getitem__(self, index):
        video, conds = self.videos[index], self.pesudo_video_clips[index]
        init_frame = random.randint(0,len(video)-self.video_length)
        video = video[init_frame:init_frame+self.video_length] # video begin with a random frame
        conds = conds[init_frame:init_frame+self.video_length]
        assert(len(video) == self.video_length)

        frames,pseudo_frames = self.load_and_transform_frames(video, conds,self.loader, self.img_transform)

        frames = torch.cat(frames, 1) # c,t,h,w
        frames = frames.transpose(0, 1) # t,c,h,w

        pseudo_frames = torch.cat(pseudo_frames, 1) # c,t,h,w
        pseudo_frames = pseudo_frames.transpose(0, 1) # t,c,h,w
        
        example = dict()
        example["pixel_values"] = frames
        example["pseudo_pixel_values"] = pseudo_frames
        example["images"],_ = self.load_and_transform_frames(video, conds,self.loader)

        
        return example



