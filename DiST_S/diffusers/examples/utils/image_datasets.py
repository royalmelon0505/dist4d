import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data as data

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import PIL
from PIL import Image
import json
import pickle as pkl
import os
import numpy as np
import random
import cv2
from torchvision import transforms as tr

from ipdb import set_trace

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]

TEXT_CONDITIONS = [
            'lighting',
            'crowdedness',
            'nusc_description',
            'nusc_description_av2_view',
            'nusc_view',
            'nusc_light_weather_av2_view',
            'nusc_light_weather_view'
        ] # indicate which conditions will be combined as a text discription

NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
CLASSES = {'car':0, 'truck':1, 'trailer':2, 'bus':3, 'construction_vehicle':4,
               'bicycle':5, 'motorcycle':6, 'pedestrian':7, 'traffic_cone':8,
               'barrier':9 }

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')

def default_loader(path):
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    '''
    return pil_loader(path)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    
def parse_comma_meta(info_path):
    with open(info_path, 'r') as f:
        meta = json.load(f)
    return meta

def parse_nusc_meta(info_path):
    with open(info_path, 'rb') as f:
        meta = pkl.load(f) # for nusc, this is a dict, convert to a list to train image LDM
    
    output_meta = []
    for scene_name, scene in meta.items():
        for frame_info in scene:
            cam_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            for cam in cam_name:
                item = dict()
                item['dataset_name'] = 'nuscenes'
                item['image_path'] = frame_info[cam]
                item['description'] = frame_info['description']
                item['cam'] = cam
                output_meta.append(item)
    
    return output_meta

def parse_nusc_meta_multiview(info_path):
    with open(info_path, 'rb') as f:
        meta = pkl.load(f) # for nusc, this is a dict, convert to a list to train image LDM
            
    output_meta = []
    for scene_name, scene in meta.items():
        for frame_info in scene:
            cam_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            item = dict()
            item['dataset_name'] = 'nuscenes'
            item['description'] = frame_info['description']
            for cam in cam_name:
                map_key = cam+'_map_rgb'
                if map_key in frame_info.keys():
                    item[map_key] = frame_info[map_key]
                item[cam] = frame_info[cam]
            output_meta.append(item)
    return output_meta

def parse_nusc_meta_multiview_video(info_path):
    with open(info_path, 'rb') as f:
        meta = pkl.load(f) # for nusc, this is a dict, convert to a list to train image LDM
    
    output_meta = []
    for scene_name, scene in meta.items():
        output_meta.append(scene)
    
    return output_meta

def parse_av2_meta(info_path):
    with open(info_path, 'rb') as f:
        meta = pkl.load(f) # for nusc, this is a dict, convert to a list to train image LDM
    
    output_meta = []
    for scene_name, scene in meta.items():
        for frame_info in scene:
            cam_name = ['front_center', 'front_right', 'front_left', 'side_left', 'side_right', 'rear_left', 'rear_right']
            for cam in cam_name:
                item = dict()
                item['dataset_name'] = 'av2'
                item['image_path'] = frame_info[cam]
                item['cam'] = cam
                output_meta.append(item)
    
    return output_meta

def parse_waymo_meta(info_path):
    with open(info_path, 'rb') as f:
        meta = pkl.load(f) # for nusc, this is a dict, convert to a list to train image LDM
    
    output_meta = []

    for scene_name, scene in meta.items():
        for frame_info in scene:
            # cam_name = ['front', 'front_left', 'front_right', 'back_left', 'back_right']
            cam_name = ['front_image']
            for cam in cam_name:
                item = dict()
                item['dataset_name'] = 'waymo'
                item['image_path'] = frame_info[cam]
                item['cam'] = cam
                output_meta.append(item)
    
    return output_meta
    
def nusc_description(item):
    text = ''
    text += item['description']

    cam = ''
    cam_info = item['cam'].split('_')[1:]

    if len(cam_info) == 2:
        cam_info = cam_info[0].lower() + ' ' + cam_info[1].lower() + ' view.'
    elif len(cam_info) == 1:
        cam_info = cam_info[0].lower() + ' view.'
    else:
        raise ValueError

    text = text + ', ' + cam_info
    return text

def nusc_light_weather(item):
    text = ''
    desp = item['description'].lower()

    if 'night' in desp:
        text += ' night,'
    if 'rain' in desp:
        text += ' rain,'
    
    if text.endswith(','):
        text = text[:-1] + '.'
    text = text.lstrip()

    return text

def nusc_view(item):

    cam_info = item['cam'].split('_')[1:]

    if len(cam_info) == 2:
        cam_info = cam_info[0].lower() + ' ' + cam_info[1].lower() + ' view.'
    elif len(cam_info) == 1:
        cam_info = cam_info[0].lower() + ' view.'
    else:
        raise ValueError

    return cam_info

def av2_view(item):

    assert item['dataset_name'] == 'av2'
    cam_info = item['cam'].split('_')
    assert len(cam_info) == 2
    cam_info = cam_info[0].lower() + ' ' + cam_info[1].lower() + ' view.'
    return cam_info


def nusc_description_av2_view(item):
    # Since AV2 has no description, only use view information
    if item['dataset_name'] == 'nuscenes':
        return 'nuscenes: ' + nusc_description(item)
    
    assert item['dataset_name'] == 'av2'

    cam_info = item['cam'].split('_')
    assert len(cam_info) == 2
    cam_info = cam_info[0].lower() + ' ' + cam_info[1].lower() + ' view.'

    text = 'av2: ' + cam_info

    return text

def nusc_light_weather_av2_view(item):
    # Since AV2 has no description, only use view information
    if item['dataset_name'] == 'nuscenes':
        light_weather = nusc_light_weather(item)
        if len(light_weather) > 0:
            return 'nuscenes: ' + light_weather + ' ' + nusc_view(item)
        else:
            return 'nuscenes: ' + nusc_view(item)
    
    assert item['dataset_name'] == 'av2'

    text = 'av2: ' + av2_view(item)

    return text

def nusc_light_weather_view(item):
    # Since AV2 has no description, only use view information
    assert item['dataset_name'] == 'nuscenes'
    light_weather = nusc_light_weather(item)

    if len(light_weather) > 0:
        return light_weather + ' ' + nusc_view(item)
    else:
        return nusc_view(item)

def waymo_light_weather_view(item):
    # Since AV2 has no description, only use view information
    assert item['dataset_name'] == 'waymo'
    light = item['time_of_day'].lower()
    weather = item['weather'].lower()
    view = item['cam'].replace('_', ' ')
    return f'{light}, {weather}, {view} view.'

def nusc_waymo_multihot(item):
    # Since AV2 has no description, only use view information
    words = []
    if item['dataset_name'] == 'nuscenes':

        text = nusc_light_weather_view(item)
        words.append('nuscenes')
    else:
        text = waymo_light_weather_view(item)
        words.append('waymo')

    if 'night' in text:
        words.append('night')
    elif 'dusk' in text:
        words.append('dusk')
    else:
        words.append('day')
    
    if 'rain' in text:
        words.append('rain')
    else:
        words.append('sunny')
    
    if 'front view' in text:
        words.append('front')
    elif 'front left view' in text:
        words.append('front left')
    elif 'front right view' in text:
        words.append('front right')
    elif 'back left view' in text:
        words.append('back left')
    elif 'back right view' in text:
        words.append('back right')
    elif 'back view' in text:
        words.append('back')
    else:
        print(text)
        raise ValueError

    return words

def lighting(item):
    # folder_name = "Chunk_7/99c94dc769b5d96e|2018-08-03--14-04-02"
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    path = item['image_path']

    hour_idx = path.find("--")
    hour = int(path[hour_idx + 2: hour_idx + 4])
    month = int(path[hour_idx - 5 : hour_idx - 3])
    assert 0 <= hour <= 24
    assert 1 <= month <= 12
    month_name = months[month - 1]

    text = None
    if 7 <= hour <= 19:
        text = f'It is daytime, {month_name}.'
    else:
        text = f'It is nighttime, {month_name}.'

    return text

def crowdedness(item):

    boxes = item['boxes'] #[x1, y1, x2, y2]
    long_sides = [max(b[2] - b[0], b[3] - b[2]) for b in boxes]
    assert all([l >= 0 for l in long_sides])

    long_sides = [l for l in long_sides if l > 5]

    n_cars = len(long_sides)
    n_large = sum([1 if l > 70 else 0 for l in long_sides])
    n_small = sum([1 if l < 30 else 0 for l in long_sides])
    n_medium = n_cars - n_large - n_small

    assert n_small >= 0

    text = ''

    if n_cars > 10:
        text += 'A crowded scene.'
    else:
        text += 'Not a crowded scene.'

    if 3 > n_large > 0:
        text += ' A few large cars.'
    elif n_large >= 3:
        text += ' Many large cars.'

    if 5 > n_small > 0:
        text += ' A few small cars.'
    elif n_small >= 5:
        text += ' Many small cars.'

    if 5 > n_medium > 0:
        text += ' A few medium cars.'
    elif n_medium >= 5:
        text += ' Many medium cars.'
    
    return text
            
def combine_text_conds(conditions, item):
    # first collect all kinds of conditions from item, put them in cond_dict
    cond_dict = {}
    for cond in conditions:
        assert isinstance(cond, str)
        cond_dict[cond] = eval(cond)(item) # use condition name as corresponding function name
    
    return cond_dict

def get_text_cond_tokens(init_caption, tokenizer, cond_dict):
    text_cond = init_caption

    for name in TEXT_CONDITIONS:
        if name in cond_dict:
            text_cond = text_cond + ' ' + cond_dict[name]
            if not text_cond.endswith('.'):
                text_cond += '.'
    
    assert text_cond != init_caption, 'Assuming conditions willy be update, in case of forgetting rewrite TEXT_CONDITIONS'
    
    text_cond = text_cond.lstrip()

    tokens = tokenizer(
        text_cond, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    input_ids = tokens.input_ids.squeeze(0)
    return input_ids

class ImgDataset(data.Dataset):
    def __init__(self,
        info_path,
        transforms,
        tokenizer
    ):
        self.loader = default_loader
        self.tokenizer = tokenizer
        self.transforms = transforms

        if isinstance(info_path, str):
            info_path = [info_path,]
            
        self.meta = []
        for p in info_path:
            if 'comma' in p:
                self.meta += parse_comma_meta(p)
            elif 'nusc' in p:
                self.meta += parse_nusc_meta(p)
                self.camera_captions = {'CAM_BACK': 'back camera view',
                                'CAM_FRONT': 'front camera view',
                                'CAM_FRONT_RIGHT':'front right camera view',
                                'CAM_FRONT_LEFT':'front left camera view',
                                'CAM_BACK_RIGHT':'back right camera view',
                                'CAM_BACK_LEFT':'back left camera view'
                                }
            elif 'av2' in p:
                self.meta += parse_av2_meta(p)
            elif 'waymo' in p:
                self.meta += parse_waymo_meta(p)
                self.camera_captions = {'front_image': 'A front view photo of driving scene'}
            else:
                raise NotImplementedError
        
        

    def __getitem__(self, index):
        item = self.meta[index]
        example = dict()
        fpath = item['image_path']
        img = self.loader(fpath)
        img = self.transforms(img)
        example["pixel_values"] = img

        view = self.camera_captions[item['cam']]

        text = self.tokenizer(view, max_length=self.tokenizer.model_max_length, 
            padding="max_length", truncation=True, return_tensors="pt").input_ids
        
        example["input_ids"] = text

        return example
    
    def __len__(self):
        return len(self.meta)
        

class ImgMultiviewDataset(data.Dataset):

    def __init__(self,
        info_path,
        conditions,
        init_caption, 
        transforms,
        tokenizer,
        multi_view = False,
        ):
        
        self.loader = default_loader
        self.tokenizer = tokenizer
        self.multi_view = multi_view
        self.conditions = conditions
        # img resolution
        self.img_w = 384
        
        if isinstance(info_path, str):
            info_path = [info_path,]
        
        self.meta = []
        for p in info_path:
            if 'comma' in p:
                self.meta += parse_comma_meta(p)
            elif 'nusc' in p:
                if self.multi_view:
                    self.meta += parse_nusc_meta_multiview(p)
                else:
                    self.meta += parse_nusc_meta(p)
            elif 'av2' in p:
                self.meta += parse_av2_meta(p)
            elif 'waymo' in p:
                self.meta += parse_waymo_meta(p)
            else:
                raise NotImplementedError
              
        assert isinstance(self.meta, list)

        self.transforms = transforms
        # convnext default transform
        self.cond_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.init_caption = init_caption 
        if not self.init_caption.endswith('.') and len(init_caption) > 0:
            self.init_caption += '.'

    def __getitem__(self, index):

        item = self.meta[index]
        example = dict()

        if self.multi_view:
            cam_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            img_list = []
            map_list = []
            box_list = []

            for cam in cam_name:
                fpath = item[cam]
                img = self.loader(fpath)
                img = self.transforms(img)
                img_list.append(img)

                if 'map' in self.conditions:
                    map_name = cam+'_map_rgb'
                    fpath_map = item[map_name]
                    map_img = self.loader(fpath_map)
                    map_img = self.cond_transforms(map_img)
                    map_list.append(map_img)
                if 'box' in self.conditions:
                    assert fpath_map is not None
                    fpath_box = fpath_map.replace('map_2d_rgb.jpg','box.png')
                    box_img = self.loader(fpath_box)
                    crop_img = box_img.crop([0,100,box_img.size[0],box_img.size[1]])
                    crop_img = crop_img.resize((self.img_w,self.img_w//2))
                    box_img = self.cond_transforms(crop_img)
                    box_list.append(box_img)
                    # crop_img.save('box.png') # for debug
                    
            imgs = torch.stack(img_list)
            if 'map' in self.conditions:
                maps = torch.stack(map_list)
                example["map_values"] = maps
            if 'box' in self.conditions:
                boxs = torch.stack(box_list)
                example["box_values"] = boxs
        
            example["pixel_values"] = imgs

        else:
            raise NotImplementedError
        
        return example
    
    def box_processing(self,box_info):
        box_result = []
        for box in box_info:
            name = box_info[0]['name']
            if name not in NameMapping.keys():
                continue
            else:
                label = CLASSES[NameMapping[name]]
            corners = np.array(box['corners'])
            # 0,x 1,y
            corners[:,0] = corners[:,0] / 1600
            corners[:,1] = (corners[:,1] - 100) / 800
            yaw = box['orientation']
            corners = corners.reshape(-1)
            box_result.append([label,yaw,*corners])
        box_result = torch.tensor(box_result)
        if len(box_result)>self.Nb:
            box_result = box_result[:self.Nb]
        else:
            box_result = torch.cat([box_result,torch.zeros(self.Nb-len(box_result),18)],dim=0)

        return box_result

    
    def get_multihot_condition(self, cond_dict):

        assert len(self.conditions) == 1
        words = cond_dict[self.conditions[0]]
        tokens = self.multihot_tokenizer(words).float()
        return tokens
    
    def parse_nusc_sweeps_meta(self, sweep_path):
        files = sorted(os.listdir(sweep_path))
        output_meta = []

        for file in files:
            item = {}
            item['dataset_name'] = 'nuscenes'
            item['image_path'] = os.path.join(sweep_path,file)
            item['description'] = ''
            item['cam'] = file.split('__')[1]
            output_meta.append(item)
        
        return output_meta

    def get_box_condition(self, cond_dict):
        raise NotImplementedError

    def __len__(self):
        return len(self.meta)

class SingleFrameDataset(data.Dataset):

    def __init__(self,
        info_path,
        conditions,
        init_caption, # 与数据集无关的描述，默认是空
        transforms,
        tokenizer,
        multihot_tokenizer,
        sweeps_path = None,
        with_box = False,
        with_cam = True,
        multi_view = False,
        do_text_condition=True,
        do_multihot_condition=False,
        do_box_condition=False,
        using_seg=False,
        using_bev=False,
        using_map=False,
        using_box=False,
        ):
        
        self.loader = default_loader
        self.tokenizer = tokenizer
        self.multihot_tokenizer = multihot_tokenizer
        self.do_multihot_condition = do_multihot_condition
        self.do_text_condition = do_text_condition
        self.do_box_condition = do_box_condition
        self.multi_view = multi_view
        self.with_box = with_box
        self.with_cam = with_cam
        self.sweeps_path = sweeps_path
        self.using_seg = using_seg
        self.using_bev = using_bev
        self.using_map = using_map
        self.using_box = using_box
        self.box_gligen = False
        self.Nb = 32

        # img resolution
        self.img_w = 512

        if isinstance(info_path, str):
            info_path = [info_path,]
        
        self.meta = []
        for p in info_path:
            if 'comma' in p:
                self.meta += parse_comma_meta(p)
            elif 'nusc' in p:
                if self.multi_view:
                    self.meta += parse_nusc_meta_multiview(p)
                else:
                    self.meta += parse_nusc_meta(p)
            elif 'av2' in p:
                self.meta += parse_av2_meta(p)
            elif 'waymo' in p:
                self.meta += parse_waymo_meta(p)
            else:
                raise NotImplementedError
        
        if self.sweeps_path is not None:
            self.meta += self.parse_nusc_sweeps_meta(self.sweeps_path)
              
        assert isinstance(self.meta, list)

        self.transforms = transforms
        # convnext default transform
        self.cond_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.conditions = conditions

        for cond in conditions:
            assert cond in globals(), f'No implementation for condition {cond}'
        
        self.init_caption = init_caption 
        if not self.init_caption.endswith('.') and len(init_caption) > 0:
            self.init_caption += '.'

    def __getitem__(self, index):

        item = self.meta[index]
        example = dict()

        if self.multi_view:
            cam_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            img_list = []
            map_list = []
            bev_list = []
            box_list = []

            if self.with_cam:
                cam_list = []
                camera_json = item['camera_info']

                with open(camera_json, 'r') as f:
                    cam_info = json.load(f)

            for cam in cam_name:
                fpath = item[cam]
                img = self.loader(fpath)
                img = self.transforms(img)
                img_list.append(img)

                if self.with_cam:

                    Extrinsic = Quaternion(cam_info[cam]["rotation"]).transformation_matrix
                    Extrinsic[:3, 3] = cam_info[cam]['translation']
                    Extrinsic = torch.tensor(Extrinsic)

                    Intrinsic = torch.tensor(cam_info[cam]['camera_intrinsic'])
                    ##  T,S for camera intrinsic fix
                    s = self.img_w/1600
                    T = torch.tensor([[1,0,0],[0,1,-100.0],[0,0,1]])
                    S = torch.tensor([[s,0,0],[0,s,0],[0,0,1]])
                    Intrinsic = torch.matmul(S,torch.matmul(T,Intrinsic))
                    cam_EI = torch.cat([Extrinsic.reshape(-1), Intrinsic.reshape(-1),torch.inverse(Extrinsic).reshape(-1),torch.inverse(Intrinsic).reshape(-1)], axis=0)
                    cam_list.append(cam_EI)

                map_name = cam+'_map_rgb'
                if map_name in item.keys():
                    fpath_map = item[map_name]
                    if self.using_box:
                        
                        if self.box_gligen:
                            fpath_box = fpath_map.replace('map_2d_rgb.jpg','box.json')
                            with open(fpath_box, 'r') as f:
                                box_info = json.load(f)
                                f.close()
                            box_img = self.box_processing(box_info)
                        else:
                            fpath_box = fpath_map.replace('map_2d_rgb.jpg','box.png')
                            fpath_box = fpath_box.replace('nusc_512256_map','nusc_512256_box')
                            box_img_class = []
                            for i in range(10):
                                fpath_box_i = fpath_box.replace('box.png',f'box{i}.png')
                                box_img_i = self.loader(fpath_box_i)
                                crop_img_i = box_img_i.crop([0,100,box_img_i.size[0],box_img_i.size[1]])
                                crop_img_i = crop_img_i.resize((self.img_w,self.img_w//2))
                                crop_img_i = self.cond_transforms(crop_img_i)
                                box_img_class.append(crop_img_i)
                            box_img = torch.stack(box_img_class)
                            # box_img = self.loader(fpath_box)
                            # crop_img = box_img.crop([0,100,box_img.size[0],box_img.size[1]])
                            # crop_img = crop_img.resize((self.img_w,self.img_w//2))
                            # for debug
                            # crop_img.save('box.png')
                            # box_img = self.cond_transforms(crop_img)
                        box_list.append(box_img)
                    if self.using_map:
                        map_img = self.loader(fpath_map)
                        map_img = self.cond_transforms(map_img)
                    if self.using_bev:
                        name = fpath_map.split('/')[-1]
                        fpath_bev = fpath_map.replace(name,'bev_seg.png')
                        bev_img = self.loader(fpath_bev)
                        bev_img = self.cond_transforms(bev_img)
                    if self.using_seg:
                        fpath_map = fpath_map.replace('_map_2d_rgb','_seg')
                    if self.with_box:
                        box_name = cam+'_box'
                        fpath_box = fpath_map.replace(map_name,box_name)
                        box_img = self.loader(fpath_box)
                        box_img = self.transforms(box_img)
                        box_list.append(box_img)
                    map_list.append(map_img)
                    bev_list.append(bev_img)
                else:
                    blank_image = Image.new("RGB", (img.shape[-1], img.shape[-2]), (255,255,255))
                    map_img = self.transforms(blank_image)
                    map_list.append(map_img)
            imgs = torch.stack(img_list)
            maps = torch.stack(map_list)
            bevs = torch.stack(bev_list)
            if self.using_box:
                boxs = torch.stack(box_list)
                example["box_values"] = boxs
            if self.with_cam:
                cams = torch.stack(cam_list)

            example["pixel_values"] = imgs
            example["map_values"] = maps
            example["cam_values"] = cams
            example["bev_values"] = bevs

            # text_list = []
            # for view in ['front view.','front right view.', 'back right view.', 'back view.','back left view.','front left view.']:
            #     desp = item['description'].lower()
            #     if 'night' in desp:
            #         view = view +' night.'
            #     else:
            #         view = view +' day.'
            #     if 'rain' in desp:
            #         view = view +' rain.'
            #     else:
            #         view = view +' sunny.'
            #     text_list.append(self.tokenizer(
            #         view, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            #     ).input_ids)

            # texts = torch.stack(text_list)
            # example["input_ids"] = texts

        else:
            # make clip tensor
            fpath = item['image_path']
            img = self.loader(fpath)
            img = self.transforms(img)

            example["pixel_values"] = img

            cond_dict = combine_text_conds(self.conditions, item)
            
            # get the formated text from cond_dict as final text condition
            if self.do_text_condition:
                example['input_ids'] = get_text_cond_tokens(self.init_caption, self.tokenizer, cond_dict)
            
            if self.do_box_condition:
                example['box_condition'] = self.get_box_condition(cond_dict)
            
            if self.do_multihot_condition:
                assert not self.do_text_condition
                example['input_ids'] = self.get_multihot_condition(cond_dict)
        
        return example
    
    def box_processing(self,box_info):
        box_result = []
        for box in box_info:
            name = box_info[0]['name']
            if name not in NameMapping.keys():
                continue
            else:
                label = CLASSES[NameMapping[name]]
            corners = np.array(box['corners'])
            # 0,x 1,y
            corners[:,0] = corners[:,0] / 1600
            corners[:,1] = (corners[:,1] - 100) / 800
            yaw = box['orientation']
            corners = corners.reshape(-1)
            box_result.append([label,yaw,*corners])
        box_result = torch.tensor(box_result)
        if len(box_result)>self.Nb:
            box_result = box_result[:self.Nb]
        else:
            box_result = torch.cat([box_result,torch.zeros(self.Nb-len(box_result),18)],dim=0)

        return box_result

    
    def get_multihot_condition(self, cond_dict):

        assert len(self.conditions) == 1
        words = cond_dict[self.conditions[0]]
        tokens = self.multihot_tokenizer(words).float()
        return tokens
    
    def parse_nusc_sweeps_meta(self, sweep_path):
        files = sorted(os.listdir(sweep_path))
        output_meta = []

        for file in files:
            item = {}
            item['dataset_name'] = 'nuscenes'
            item['image_path'] = os.path.join(sweep_path,file)
            item['description'] = ''
            item['cam'] = file.split('__')[1]
            output_meta.append(item)
        
        return output_meta

    def get_box_condition(self, cond_dict):
        raise NotImplementedError

    def __len__(self):
        return len(self.meta)

def get_multi_condition_image_loader(
        info_path, conditions, init_caption, transforms, shuffle, batch_size, num_workers, drop_last,
        tokenizer=None,
        multihot_tokenizer=None,
        multihot_condition=False,
        box_condition=False,
        text_condition=True,
    ):


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    dataset = SingleFrameDataset(
        info_path, 
        conditions,
        init_caption,
        transforms,
        tokenizer,
        multihot_tokenizer,
        do_multihot_condition=multihot_condition,
        do_box_condition=box_condition,
        do_text_condition=text_condition,
    )

    # DataLoaders creation:
    # NOTE: drop last to make loss curve smoother
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return dataloader

class SingleFrameUpscaleDataset(data.Dataset):
    
    def __init__(self,
        low_res_path,
        high_res_path,
        tokenizer,
        transforms,
        do_text_condition=True,
        ):
        self.loader = default_loader
        self.tokenizer = tokenizer
        self.low_res_path = low_res_path
        self.high_res_path = high_res_path
        self.do_text_condition = do_text_condition
        self.transforms = transforms

        self.files = self.parse_nusc_info()

        self.camera_captions = {'CAM_BACK': 'A photo of driving scene from back camera view',
                                'CAM_FRONT': 'A photo of driving scene from front camera view',
                                'CAM_FRONT_RIGHT':'A photo of driving scene from front right camera view',
                                'CAM_FRONT_LEFT':'A photo of driving scene from front left camera view',
                                'CAM_BACK_RIGHT':'A photo of driving scene from back right camera view',
                                'CAM_BACK_LEFT':'A photo of driving scene from back left camera view'
                                }

    
    def __len__(self):
        return len(self.files)
    
    def parse_nusc_info(self):

        with open(self.low_res_path, 'rb') as f1:
            low_res_info = pkl.load(f1)
        with open(self.high_res_path, 'rb') as f2:
            high_res_info = pkl.load(f2)
        
        assert len(low_res_info)==len(high_res_info)

        output_files = []
        
        for low_key, high_key in zip(low_res_info, high_res_info):
            assert low_key==high_key
            low_scene = low_res_info[low_key]
            hign_scene = high_res_info[high_key]
            for frame_i in range(len(low_scene)):
                frame_info = low_scene[frame_i]
                frame_info_high = hign_scene[frame_i]
                assert frame_info['frame_id']==frame_info_high['frame_id']
                cam_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                for cam in cam_name:
                    item = dict()
                    item['dataset_name'] = 'nuscenes'
                    item['image_path'] = frame_info[cam]
                    item['high_image_path'] = frame_info_high[cam]
                    item['description'] = frame_info['description']
                    item['cam'] = cam
                    output_files.append(item)
        return output_files
    
    def __getitem__(self, index):

        item = self.files[index]
        # make clip tensor
        low_path = item['image_path']
        high_path = item['high_image_path']
        img = self.loader(low_path)
        high_img = self.loader(high_path)

        img = self.transforms(img)
        high_img = self.transforms(high_img)

        example = dict()
        example["low_res_pixel_values"] = img
        example["high_res_pixel_values"] = high_img

        cam_type = item['cam']
        text = self.camera_captions[cam_type]
        

        example["input_ids"] = self.tokenizer(
                    text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids

        return example

class MultiFrameDataset(data.Dataset):

    def __init__(self,
        info_path,
        conditions,
        init_caption, # 与数据集无关的描述，默认是空
        transforms,
        tokenizer,
        multihot_tokenizer,
        sweeps_path = None,
        with_box = False,
        with_cam = True,
        multi_view = False,
        do_text_condition=True,
        do_multihot_condition=False,
        do_box_condition=False,
        using_seg=False,
        using_bev=False,
        n_frames  = 3,
        ):
        
        self.loader = default_loader
        self.tokenizer = tokenizer
        self.multihot_tokenizer = multihot_tokenizer
        self.do_multihot_condition = do_multihot_condition
        self.do_text_condition = do_text_condition
        self.do_box_condition = do_box_condition
        self.multi_view = multi_view
        self.with_box = with_box
        self.with_cam = with_cam
        self.sweeps_path = sweeps_path
        self.using_seg = using_seg
        self.using_bev = using_bev
        self.n_frames = n_frames

        if isinstance(info_path, str):
            info_path = [info_path,]
        
        self.meta = []
        for p in info_path:
            if 'comma' in p:
                self.meta += parse_comma_meta(p)
            elif 'nusc' in p:
                if self.multi_view:
                    self.meta += parse_nusc_meta_multiview_video(p)
                else:
                    self.meta += parse_nusc_meta(p)
            elif 'av2' in p:
                self.meta += parse_av2_meta(p)
            elif 'waymo' in p:
                self.meta += parse_waymo_meta(p)
            else:
                raise NotImplementedError
        
        if self.sweeps_path is not None:
            self.meta += self.parse_nusc_sweeps_meta(self.sweeps_path)
              
        assert isinstance(self.meta, list)

        self.transforms = transforms
        self.conditions = conditions

        for cond in conditions:
            assert cond in globals(), f'No implementation for condition {cond}'
        
        self.init_caption = init_caption 
        if not self.init_caption.endswith('.') and len(init_caption) > 0:
            self.init_caption += '.'


    def __getitem__(self, index):

        video = self.meta[index]
        init_frame = random.randint(0,len(video)-self.n_frames)
        video = video[init_frame:init_frame+self.n_frames] # video begin with a random frame
        assert(len(video) == self.n_frames)

        example = dict()

        video_list = []
        video_seg_list = []
        video_cam_list = []
        for frame in video:
            cam_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            img_list = []
            map_list = []
            if self.with_cam:
                cam_list = []
                camera_json = frame['camera_info']
                with open(camera_json, 'r') as f:
                    cam_info = json.load(f)
            if self.with_box:
                box_list = []
            for cam in cam_name:
                fpath = frame[cam]
                img = self.loader(fpath)
                img = self.transforms(img)
                img_list.append(img)

                if self.with_cam:
                    Extrinsic = Quaternion(cam_info[cam]["rotation"]).transformation_matrix
                    Extrinsic[:3, 3] = cam_info[cam]['translation']
                    Extrinsic = torch.tensor(Extrinsic)
                    Intrinsic = torch.tensor(cam_info[cam]['camera_intrinsic'])
                    cam_EI = torch.cat([Extrinsic.reshape(-1), Intrinsic.reshape(-1)], axis=0)
                    cam_list.append(cam_EI)

                map_name = cam+'_map_rgb'
                if map_name in frame.keys():
                    fpath_map = frame[map_name]
                    if self.using_bev:
                        name = fpath_map.split('/')[-1]
                        fpath_map = fpath_map.replace(name,'bev_seg.png')
                    if self.using_seg:
                        fpath_map = fpath_map.replace('_map_2d_rgb','_seg')
                    map_img = self.loader(fpath_map)
                    map_img = self.transforms(map_img)
                    if self.with_box:
                        box_name = cam+'_box'
                        fpath_box = fpath_map.replace(map_name,box_name)
                        box_img = self.loader(fpath_box)
                        box_img = self.transforms(box_img)
                        box_list.append(box_img)
                    map_list.append(map_img)
                else:
                    blank_image = Image.new("RGB", (img.shape[-1], img.shape[-2]), (255,255,255))
                    map_img = self.transforms(blank_image)
                    map_list.append(map_img)
            imgs = torch.stack(img_list)
            maps = torch.stack(map_list)
            if self.with_box:
                boxs = torch.stack(box_list)
                example["box_values"] = boxs
            if self.with_cam:
                cams = torch.stack(cam_list)
            video_list.append(imgs)
            video_seg_list.append(maps)
            video_cam_list.append(cams)
        videos = torch.stack(video_list)
        videos_seg = torch.stack(video_seg_list)
        videos_cam = torch.stack(video_cam_list)

        example["pixel_values"] = videos
        example["map_values"] = videos_seg
        example["cam_values"] = videos_cam

        text = self.init_caption
        
        example["input_ids"] = self.tokenizer(
                text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        
        return example
    
    def get_multihot_condition(self, cond_dict):

        assert len(self.conditions) == 1
        words = cond_dict[self.conditions[0]]
        tokens = self.multihot_tokenizer(words).float()
        return tokens
    
    def parse_nusc_sweeps_meta(self, sweep_path):
        files = sorted(os.listdir(sweep_path))
        output_meta = []

        for file in files:
            item = {}
            item['dataset_name'] = 'nuscenes'
            item['image_path'] = os.path.join(sweep_path,file)
            item['description'] = ''
            item['cam'] = file.split('__')[1]
            output_meta.append(item)
        
        return output_meta

    def get_box_condition(self, cond_dict):
        raise NotImplementedError

    def __len__(self):
        return len(self.meta)