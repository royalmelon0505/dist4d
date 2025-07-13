import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from .convnext import convnext_tiny,convnext_base,convnext_supertiny
from .swin_transformer import SwinTransformer
import torch.nn.functional as F
import random
from ipdb import set_trace

class LayoutCondEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, img_shape=(960,384), out_dim=1024, in_chans=3,condition_list=None):
        super().__init__()
        
        self.resize_input_width = img_shape[0]
        self.resize_input_height = img_shape[1]
        self.out_dim = out_dim
        self.conditions = condition_list
        self.down_factor = 32 # determined by the convnext backbone 
        convnext_feature_dim = 768

        assert self.resize_input_height % self.down_factor == 0
        assert self.resize_input_width % self.down_factor == 0

        # self.convnext_tiny_backbone = convnext_tiny(pretrained=True)

        self.convnext_tiny_backbone = convnext_supertiny(in_chans=in_chans,depths=[3, 3, 9, 3], dims=[32, 64, 128, 256])

        ## self.convnext_base_backbone = convnext_base(pretrained=False)

        # self.convnext_tiny_backbone = SwinTransformer()
        # self.convnext_tiny_backbone.init_weights()

        # self.final_conv=nn.Conv2d(dims[-1], 4, kernel_size=2, stride=1,padding=0)
        
        # self.num_tokens = (self.resize_input_height // self.down_factor) * (self.resize_input_width // self.down_factor)


    def forward(self, box_info=None):
        B = box_info.shape[0] 
        
        if box_info is not None:
            objs_box = self.convnext_tiny_backbone(box_info)
            objs = objs_box

      
        return objs

class LayoutCondEncoder_withref(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, img_shape=(960,384), out_dim=1024, in_chans=3,condition_list=None):
        super().__init__()
        
        self.resize_input_width = img_shape[0]
        self.resize_input_height = img_shape[1]
        self.out_dim = out_dim
        self.conditions = condition_list
        self.down_factor = 32 # determined by the convnext backbone 
        convnext_feature_dim = 768

        assert self.resize_input_height % self.down_factor == 0
        assert self.resize_input_width % self.down_factor == 0

        # self.convnext_tiny_backbone = convnext_tiny(pretrained=True)

        self.convnext_tiny_backbone = convnext_supertiny(in_chans=in_chans,depths=[3, 3, 9, 3], dims=[32, 64, 128, 256])

        ## self.convnext_base_backbone = convnext_base(pretrained=False)

        # self.convnext_tiny_backbone = SwinTransformer()
        # self.convnext_tiny_backbone.init_weights()

        # self.final_conv=nn.Conv2d(8, 4, kernel_size=1, stride=1,padding=0)
        
        # self.num_tokens = (self.resize_input_height // self.down_factor) * (self.resize_input_width // self.down_factor)
        

    def forward(self, box_info=None,ref_latent = None):
        B = box_info.shape[0] 
        
        if box_info is not None:
            pseudo_latent = self.convnext_tiny_backbone(box_info)

        # pseudo_latent = torch.cat([pseudo_latent,ref_latent[0].repeat([pseudo_latent.shape[0],1,1,1])],axis=1)
        # pseudo_latent = self.final_conv(pseudo_latent)

        return pseudo_latent

class EgoCondEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, out_dim=1024):
        super().__init__()
        self.out_dim = out_dim

        self.linears = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU())
    
    def forward(self, ego):
        ego_latents = self.linears(ego) + self.shortcut(ego)
        return ego_latents

if __name__=='__main__':
    temp = SecondEncoder()
    temp.save_pretrained()