#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import random

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data as data
import transformers
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler,EulerDiscreteScheduler

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
# from ipdb import set_trace
from torchvision import transforms

# ours code base
# from utils.video_datasets import VideoNuscenesDataset, Video_reproj_nus_Dataset
from utils.video_datasets_mm import Video_reproj_nus_multi_model_Dataset
from src.encoder.unified_encoder import LayoutCondEncoder
from src.model.unet_spatial_temporal_condition_custom import UNetSpatioTemporalConditionModelV2, UNetSpatioTemporalConditionModelV2_inchannel16
from src.pipelines.pipeline_stable_video_diffusion_custom import StableVideoDiffusionPipeline_convnb
from src.pipelines.pipeline_stable_video_diffusion_custom_mm import StableVideoDiffusionPipeline_convnb_mm
from diffusers import StableVideoDiffusionPipeline
from diffusers.models import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPFeatureExtractor,
)
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from diffusers.utils.torch_utils import randn_tensor

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate import FullyShardedDataParallelPlugin

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def init_model_from_config_and_load_pretrain(cls, config_path, model_file, ignore_mismatched_sizes=False, torch_dtype=None, freeze=False):
    config = cls.load_config(config_path)
    model = cls.from_config(config)

    from safetensors import safe_open

    state_dict = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # state_dict = torch.load(model_file, map_location="cpu")
    model._convert_deprecated_attention_blocks(state_dict)

    model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
        model,
        state_dict,
        model_file, # not used in the function
        model_file,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )

    if freeze:
        update_param_list = []
        for param_name, param in model.named_parameters():
            if param_name in missing_keys:
                update_param_list.append(param_name)
                param.requires_grad_(True)
            else:
                if 'attn' in param_name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        assert len(update_param_list)==len(missing_keys)

    loading_info = {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
        "error_msgs": error_msgs,
    }

    if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
        raise ValueError(
            f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
        )
    elif torch_dtype is not None:
        model = model.to(torch_dtype)

    # not ensure what is this
    # model.register_to_config(_name_or_path=pretrained_model_name_or_path)

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    return model

def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

# def log_validation(vae, condition_encoder, unet, args, accelerator, weight_dtype, epoch):
#     logger.info("Running validation... ")

#     pipeline = StableVideoDiffusionPipeline_convnb_mm.from_pretrained(
#         args.pretrained_model_name_or_path,
#         vae=accelerator.unwrap_model(vae),
#         layout_encoder=accelerator.unwrap_model(condition_encoder),
#         unet=accelerator.unwrap_model(unet),
#         safety_checker=None,
#         revision=args.revision,
#         torch_dtype=weight_dtype,
#     )
#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     if args.enable_xformers_memory_efficient_attention:
#         pipeline.enable_xformers_memory_efficient_attention()

#     if args.seed is None:
#         generator = None
#     else:
#         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

#     images = []
#     for i in range(len(args.validation_prompts)):
#         with torch.autocast("cuda"):
#             image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

#         images.append(image)

#     for tracker in accelerator.trackers:
#         if tracker.name == "tensorboard":
#             np_images = np.stack([np.asarray(img) for img in images])
#             tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
#         elif tracker.name == "wandb":
#             tracker.log(
#                 {
#                     "validation": [
#                         wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
#                         for i, image in enumerate(images)
#                     ]
#                 }
#             )
#         else:
#             logger.warn(f"image logging not implemented for {tracker.name}")

#     del pipeline
#     torch.cuda.empty_cache()

#     return images

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--random_frame_mask",
        default=False,
        action="store_true",
        help="random_frame_mask or fix masked frame number.",
    )
    parser.add_argument(
        "--single_cam",
        default=False,
        action="store_true",
        help="single_cam or multi_cam.",
    )
    parser.add_argument(
        "--with_valid_mask",
        default=False,
        action="store_true",
        help="with_valid_mask in depth 3rd channel or not",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Number of interval frames.",
    )
    parser.add_argument(
        "--given_frame_num",
        type=int,
        default=None,
        help="If fix masked frame number, the number of it.",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=10,
        help="Number of video frames.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=320,
        help="Target height of resized video frames.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=512,
        help="Target width of resized video frames.",
    )
    parser.add_argument(
        "--video_top_crop",
        type=int,
        default=100,
        help="Top crop of video frames. Used to remove the sky.",
    )
    parser.add_argument(
        "--video_center_crop",
        action="store_true",
        help="Center crop of video frames.",
    )
    # parser.add_argument(
    #     "--camera_view",
    #     type=str,
    #     nargs="+",
    #     required=True,
    #     help="Camera view to use. Set according to your pickle file format.",
    # )
    parser.add_argument(
        "--conditions",
        type=str,
        default='',
        help="conditions for training.",
    )
    parser.add_argument(
        "--use_svd_pretrain",
        default=False,
        action="store_true",
        help="whether use svd pretrain.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--seed_for_gen", type=int, default=42, help="A seed for generator.")

    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--initialize_pseudoimg_encoder",
        action="store_true",
        default=False,
        help="initialize_pseudoimg_encoder from scratch.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--visualize_data_batch",
        default=False,
        action="store_true",
        help="Whether to visualize the data batch.",
    )
    parser.add_argument(
        "--load_from_disk",
        default=False,
        action="store_true",
        help=(
            "Whether to from prepared parquet data."
        ),
    )
    parser.add_argument(
        "--load_from_pkl",
        default=False,
        action="store_true",
        help=(
            "Whether to from prepared pkl data."
        ),
    )
    parser.add_argument(
        "--load_from_list",
        default=False,
        action="store_true",
        help="The path to the list of files to load.",
    )
    parser.add_argument(
        "--ego",
        default=False,
        action="store_true",
        help=(
            "Whether to cond on ego info."
        ),
    )
    parser.add_argument(
        "--fixed_caption",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--unet_pretrained_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--unet_config_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--old_loading",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--scheduler_config_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ddim",
        default=True,
        action="store_true",
        help=(
            "Whether to DDIM."
        ),
    )
    parser.add_argument(
        "--freeze_pretrain",
        default=False,
        action="store_true",
        help=(
            "Whether to freeze single image pretrain params."
        ),
    )
    parser.add_argument(
        "--do_mode_vae_encode_for_context_frame",
        default=False,
        action="store_true",
        help=(
            "Use mode instead of sampling when encode context frame for training. A trick from pix2pix."
        )
    )
    parser.add_argument(
        "--init_caption",
        type=str,
        default='',
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=7, # svd
    )

    parser.add_argument(
        "--mismatch_aug_ratio",
        type=float,
        default=None,
        help="mismatch_aug_ratio",
    )
    parser.add_argument(
        "--use_spatial_tokens_in_condition",
        action="store_true",
        help="Whether to use spatial tokens in the conditioning.",
    )
    parser.add_argument(
        "--use_vq_token_in_context",
        action="store_true",
        help="Whether to use VQ tokens in the context.",
    )
    parser.add_argument(
        "--add_vq_feature_to_latent",
        action="store_true",
        help="Whether to use additive VQ features.",
    )
    parser.add_argument(
        "--cat_vq_feature_to_latent",
        action="store_true",
        help="Whether to concatenate VQ features.",
    )
    parser.add_argument(
        "--vq_image_width",
        type=int,
        default=None,
        help="The width of the VQ encoded image.",
    )
    parser.add_argument(
        "--vq_image_height",
        type=int,
        default=None,
        help="The height of the VQ encoded image.",
    )
    parser.add_argument(
        "--vq_model_type",
        type=str,
        choices=["VQ-16", "VQ-8"],
        default="VQ-8",
        help="The type of VQ model to use.",
    )
    parser.add_argument(
        "--vq_model_ckpt",
        type=str,
        default="pretrained/vq_ds8_c2i.pt",
        help="The path to the VQ model checkpoint.",
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


# ================================================================
# condition: sparse RGB + sparse depth
# output: dense RGB + dense depth
# ================================================================
def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=FullyShardedDataParallelPlugin(activation_checkpointing=False),
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed_for_gen)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # move accelerate init here
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        # tracker_config["camera_view"] = tracker_config["camera_view"][0]
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Load scheduler and models.
    Diff_Scheduler = EulerDiscreteScheduler

    if args.scheduler_config_path is not None:
        scheduler_config = Diff_Scheduler.load_config(args.scheduler_config_path)
        noise_scheduler = Diff_Scheduler.from_config(scheduler_config)
        logger.info(f'******** Using scheduler config: {args.scheduler_config_path} ********')
        logger.info(f'******** Scheduler config: {scheduler_config} ********')
    else:
        logger.info('Using default scheduler from pretrain')
        noise_scheduler = Diff_Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    class PackedModule(torch.nn.Module):
        def __init__(self, unet, layout_encoder, layout_encoder_depth, conditioning_dropout_prob):
            super().__init__()
            self.unet = unet
            self.layout_encoder = layout_encoder
            self.layout_encoder_depth = layout_encoder_depth
            self.conditioning_dropout_prob = conditioning_dropout_prob

        def forward(self,
                    first_frame_image,
                    first_frame_depth_image,
                    encoder_hidden_states,
                    noisy_latents,
                    c_noise,
                    add_time_ids,
                    scaling_factor,
                    bsz,
                    c_in,
                    weight_dtype):


            first_frame_latent = self.layout_encoder(box_info = first_frame_image) * scaling_factor
            first_frame_depth_latent = self.layout_encoder_depth(box_info = first_frame_depth_image) * scaling_factor

            
            if args.conditioning_dropout_prob > 0: 
                random_p0 = torch.rand(bsz, device=first_frame_latent.device)
                # random_p0 = torch.zeros(bsz, device=first_frame_latent.device)
                # Sample masks for the edit prompts.
                prompt_mask = random_p0 < args.conditioning_dropout_prob
                prompt_mask = prompt_mask.reshape(bsz, 1, 1, 1) # B x 1 x 1

                # import pdb
                # pdb.set_trace()

                prompt_mask=prompt_mask.repeat(first_frame_latent.shape)

                # Random dropping for unconditional training
                first_frame_latent = torch.where(prompt_mask, torch.zeros_like(first_frame_latent), first_frame_latent)
                first_frame_depth_latent = torch.where(prompt_mask, torch.zeros_like(first_frame_depth_latent), first_frame_depth_latent)
            
            first_frame_latents = first_frame_latent.unsqueeze(0)#.repeat(1, nframes, 1, 1, 1)
            first_frame_depth_latents = first_frame_depth_latent.unsqueeze(0)

            input_latents = torch.cat([c_in * noisy_latents, first_frame_latents/scaling_factor, first_frame_depth_latents/scaling_factor], dim=2).half()
            # input_latents = torch.cat([c_in * noisy_latents, first_frame_latents, first_frame_depth_latents], dim=2).half()

            # import pdb
            # pdb.set_trace()

            # print(unet.time_embedding.linear_1.weight.shape)#torch.Size([1280, 320])

            model_pred = self.unet(input_latents.float(), c_noise, encoder_hidden_states=encoder_hidden_states.float(), added_time_ids=add_time_ids).sample

            return model_pred
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.

    unet = UNetSpatioTemporalConditionModelV2_inchannel16.from_pretrained(args.pretrained_model_name_or_path, 
                                                                          subfolder="unet", 
                                                                          in_channels=16,
                                                                          out_channels=8,
                                                                          low_cpu_mem_usage=False,
                                                                          ignore_mismatched_sizes=True,
                                                                          revision=args.non_ema_revision)
    
    if args.use_svd_pretrain:
        raw_unet = UNetSpatioTemporalConditionModelV2.from_pretrained(args.pretrained_model_name_or_path, 
                                                                            subfolder="unet", 
                                                                            revision=args.non_ema_revision)
        # NOTE: update the unet weights
        unet.update_unet_weights(raw_unet, update_conv_in=True, update_conv_out=True)
        del raw_unet

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )
    tokenizer = None
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )

        if args.initialize_pseudoimg_encoder:
            layout_encoder = LayoutCondEncoder(out_dim=1024, condition_list=None)
            layout_encoder_depth = LayoutCondEncoder(out_dim=1024, condition_list=None)
        else:
            layout_encoder = LayoutCondEncoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="layout_encoder")
            layout_encoder_depth = LayoutCondEncoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="layout_encoder_depth")
    
    model = PackedModule(unet, layout_encoder, layout_encoder_depth, args.conditioning_dropout_prob)

    if args.use_vq_token_in_context:
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = accelerator.mixed_precision
        from vq_decode import load_model, decode_to_codebook_features
        vq_model = load_model(args.vq_model_type, args.vq_model_ckpt, 16384, 8)
        vq_model = vq_model.to(accelerator.device, dtype=weight_dtype)

    if args.use_vq_token_in_context and args.cat_vq_feature_to_latent:
        def add_input_channels_conv(target, additional_in_channels):
            """
            Adds additional input channels to the first convolutional layer of the target model or UNet.

            Parameters:
            - target: The model or UNet object containing the convolutional layer.
            - additional_in_channels: The number of additional input channels to add.
            
            Returns:
            - target: The model or UNet with the updated convolutional layer.
            """
            
            # Determine the correct target for the convolutional layer
            if hasattr(target, 'unet'):
                conv_layer = target.unet.conv_in
            else:
                conv_layer = target.conv_in
            
            # Extract the existing weights and bias
            old_weights = conv_layer.weight.data
            old_bias = conv_layer.bias.data if conv_layer.bias is not None else None
            
            # Define new input channels
            new_in_channels = conv_layer.in_channels + additional_in_channels
            
            # Create a new convolutional layer with the extra input channels
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=conv_layer.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                dilation=conv_layer.dilation,
                groups=conv_layer.groups,
                bias=(conv_layer.bias is not None)
            )
            
            # Initialize the new input channels
            new_weights = new_conv.weight.data
            new_weights[:, :conv_layer.in_channels] = old_weights
            nn.init.kaiming_normal_(new_weights[conv_layer.in_channels:])
            
            # If there's a bias, copy the old bias to the new layer
            if old_bias is not None:
                new_conv.bias.data = old_bias
            
            # Replace the old layer with the new layer in the model or UNet
            if hasattr(target, 'unet'):
                target.unet.conv_in = new_conv
            else:
                target.conv_in = new_conv
            
            return target

        model = add_input_channels_conv(model, 8)

    # Freeze 
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    model.layout_encoder.requires_grad_(True)
    model.layout_encoder_depth.requires_grad_(True)
    model.layout_encoder.convnext_tiny_backbone.requires_grad_(True)
    model.layout_encoder_depth.convnext_tiny_backbone.requires_grad_(True)
    model.unet.requires_grad_(True)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNetSpatioTemporalConditionModelV2_inchannel16.from_pretrained(args.pretrained_model_name_or_path,
                                                                                  subfolder="unet",
                                                                                  in_channels=16,
                                                                                    out_channels=8,
                                                                                    low_cpu_mem_usage=False,
                                                                                    ignore_mismatched_sizes=True,
                                                                                  revision=args.revision)
        # ema_unet = add_input_channels_conv(ema_unet, 8)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNetSpatioTemporalConditionModelV2_inchannel16, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            all_models = ['unet', 'image_encoder', 'layout_encoder', 'layout_encoder_depth']
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, all_models[i]))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            # bug here, fix in the future
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModelV2_inchannel16)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if i==3:
                    load_model = UNetSpatioTemporalConditionModelV2_inchannel16.from_pretrained(input_dir, subfolder="unet",
                                                                                                # in_channels=16,
                                                                                                # out_channels=8,
                                                                                                # low_cpu_mem_usage=False,
                                                                                                # ignore_mismatched_sizes=True,
                                                                                                )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif i==2:
                    load_model = CLIPVisionModelWithProjection.from_pretrained(input_dir, subfolder="image_encoder")
                    # model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif i==1:
                    load_model = LayoutCondEncoder.from_pretrained(input_dir, subfolder="layout_encoder")
                    
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif i==0:
                    load_model = LayoutCondEncoder.from_pretrained(input_dir, subfolder="layout_encoder_depth")
                    
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Initialize the optimizer
    all_parameters = list(unet.parameters())+list(layout_encoder.parameters())+list(layout_encoder_depth.parameters())
    optimizer = optimizer_cls(
        all_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.load_from_pkl:
            if accelerator.is_main_process:
                print('************** Loading Dataset From PKL **************')
            # train_dataset = VideoNuscenesDataset(
            #     data_root=args.dataset_name,
            #     video_transforms=train_transforms,
            #     tokenizer=tokenizer,
            #     video_length=args.nframes,
            #     mismatch_aug_ratio=args.mismatch_aug_ratio
            # )
            pesudo_flag_name_list = ["frame_pesudo-2_mod","frame_pesudo+2_mod","cycle_reproj"]
            train_dataset = Video_reproj_nus_multi_model_Dataset(
                data_root=args.dataset_name,
                video_transforms=train_transforms,
                video_length=args.nframes,
                split_name="train",
                pesudo_flag_name_list = pesudo_flag_name_list,
                with_valid_mask=args.with_valid_mask,
                pros_w = args.video_width,
                pros_h = args.video_height,
                # mismatch_aug_ratio=args.mismatch_aug_ratio
            )
        else:
            assert False

    if accelerator.is_main_process:
        print('************** finish loading *************')

    # def tokenize_captions_null(captions):
    #     inputs = tokenizer(
    #         captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     )
    #     return inputs.input_ids
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # unet, optimizer, train_dataloader, lr_scheduler, layout_encoder = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler, layout_encoder
    # )

    if accelerator.is_main_process:
        # log model architecture
        with open(os.path.join(args.output_dir, "model_summary.txt"), "w") as f:
            f.write(str(model))

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora condition_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move the text encoder to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # layout_encoder.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    nframes = args.nframes
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            if args.use_vq_token_in_context:
                indices = batch["vq_values"]
                vq_image_height = args.vq_image_height or args.video_height
                vq_image_width = args.vq_image_width or args.video_width
                vq_codebook_features = decode_to_codebook_features(vq_model, indices, (vq_image_height, vq_image_width), args.vq_model_type)
                vq_codebook_features = torch.einsum('t h w c -> t c h w', vq_codebook_features).unsqueeze(0) # won't work for bs != 1
                if vq_image_height != args.video_height or vq_image_width != args.video_width:
                    vq_feature_height = args.vq_image_height // (8 if args.vq_model_type == "VQ-8" else 16)
                    vq_feature_width = args.vq_image_width // (8 if args.vq_model_type == "VQ-8" else 16)
                    vq_codebook_features = vq_codebook_features.view(-1, 8, vq_feature_height, vq_feature_width)
                    vq_codebook_features = F.interpolate(vq_codebook_features, (args.video_height // 8, args.video_width // 8), mode='bilinear', align_corners=False)
                    vq_codebook_features = vq_codebook_features.view(1, args.nframes, 8, args.video_height // 8, args.video_width // 8) 

            if args.visualize_data_batch:
                if accelerator.is_main_process:
                    def unnormalize_tensor(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
                        """ Un-normalize a tensor using the provided mean and std. """
                        mean = np.array(mean)[:, None, None]
                        std = np.array(std)[:, None, None]
                        tensor = tensor * std + mean
                        return tensor

                    def visualize_tensor(tensor, num_images=1, num_time_steps=1, save_path="tensor_visualization.png"):
                        """ Visualize the first few images of a tensor and save to a file. """
                        import matplotlib.pyplot as plt

                        # Un-normalize the tensor first
                        tensor = unnormalize_tensor(tensor)

                        batch_size, time_step, channel, height, width = tensor.shape

                        fig, axes = plt.subplots(num_images, num_time_steps, figsize=(num_time_steps * 4, num_images * 4))
                        print(f"axes are {axes}")
                        if num_images == 1 and num_time_steps == 1:
                            axes = [[axes]]  # Make it 2D for consistency
                        elif num_images == 1 or num_time_steps == 1:
                            axes = [axes]

                        for i in range(num_images):
                            for j in range(num_time_steps):
                                ax = axes[i][j]
                                if channel == 1:
                                    # For grayscale images, remove the channel dimension
                                    img = tensor[i, j, 0, :, :]
                                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                                else:
                                    # For RGB images
                                    img = tensor[i, j, :, :, :].transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
                                    ax.imshow(img, vmin=0, vmax=1)
                                ax.axis('off')  # Turn off axis numbers and ticks
                                ax.set_title(f"Batch {i}, Time {j}\n{height}x{width}")

                        plt.tight_layout()
                        plt.savefig(save_path)
                        plt.close()

                    print(batch["pixel_values"].shape)
                    print(batch["images"][0][0].shape)

                    visualize_tensor(batch["pixel_values"].detach().cpu().numpy(), num_images=1, num_time_steps=args.nframes, save_path="input_image.png")
                    visualize_tensor(batch["images"][0][0].transpose(0, 1).unsqueeze(1).detach().cpu().numpy(), num_images=1, num_time_steps=1, save_path="condition_image.png")

                    exit(0)

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):

                gt_latent_dist = vae.encode(batch["pixel_values"].reshape(-1, *batch["pixel_values"].shape[-3:]).to(weight_dtype)).latent_dist
                gt_latents = gt_latent_dist.sample()
                gt_latents = gt_latents * vae.config.scaling_factor
                latents = gt_latents
                latents = latents.reshape(-1, nframes, *latents.shape[-3:]) # B, T, C, H, W

                # encode depth input
                gt_depth_latent_dist = vae.encode(batch["depth_values"].reshape(-1, *batch["depth_values"].shape[-3:]).to(weight_dtype)).latent_dist
                gt_depth_latents = gt_depth_latent_dist.sample()
                gt_depth_latents = gt_depth_latents * vae.config.scaling_factor
                depth_latents = gt_depth_latents
                depth_latents = depth_latents.reshape(-1, nframes, *depth_latents.shape[-3:]) # B, T, C, H, W

                # if args.do_mode_vae_encode_for_context_frame:
                #     original_image_embeds = latent_dist.mode()
                #     original_image_embeds = original_image_embeds * vae.config.scaling_factor
                #     original_image_embeds = original_image_embeds.reshape(-1, nframes, *original_image_embeds.shape[-3:]) # B, T, C, H, W
                # else:
                #     original_image_embeds = latents.clone()

                bsz = latents.shape[0]

                # Add noise to the latents according to the noise magnitude at each timestep
                P_std = 1.6
                P_mean = 0.7
                rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=latents.device)
                sigma = (rnd_normal * P_std + P_mean).exp()
                c_skip = 1 / (sigma**2 + 1)
                c_out =  -sigma / (sigma**2 + 1) ** 0.5
                c_in = 1 / (sigma**2 + 1) ** 0.5
                c_noise = (sigma.log() / 4).reshape([bsz])
                loss_weight = (sigma ** 2 + 1) / sigma ** 2

                latents = torch.cat([latents, depth_latents], dim=2)
                noisy_latents = latents + torch.randn_like(latents) * sigma

                # condition
                # first_frame_info = batch['images'][0][0] # there is gt, we think it is useless
                #-------------------------
                # print(first_frame_info.shape)
                first_frame_info =  batch["pseudo_pixel_values"][:,0].permute(1,0,2,3)
                # print(batch["pseudo_pixel_values"].shape,first_frame_info.shape)
                # We normalize the image before resizing to match with the original implementation.
                # Then we unnormalize it after resizing.
                image = first_frame_info * 2.0 - 1.0
                image = _resize_with_antialiasing(image, (224, 224))
                image = (image + 1.0) / 2.0

                cond_transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Normalize([0.48145466,0.4578275,0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                    ]
                )

                image = image.permute(1,0,2,3)
                
                # Normalize the image with for CLIP input
                image = cond_transforms(image)
                image_embeddings = image_encoder(image).image_embeds
                image_embeddings = image_embeddings.unsqueeze(1)

                image_embeddings=image_embeddings.repeat([args.nframes,1,1])

                encoder_hidden_states = image_embeddings
                uncond_hidden_states = torch.zeros_like(image_embeddings)
        
                if args.conditioning_dropout_prob > 0: 
                    random_p0 = torch.rand(bsz, device=latents.device)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p0 < args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1) # B x 1 x 1

                    # Random dropping for unconditional training
                    encoder_hidden_states = torch.where(prompt_mask, torch.zeros_like(encoder_hidden_states), encoder_hidden_states)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)


                # keep same with the inference
                noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
                first_frame_image = batch["pseudo_pixel_values"].squeeze(0)#[:, 0]
                first_frame_image = first_frame_image + noise_aug_strength * torch.randn_like(first_frame_image)
                # first_frame_latent = vae.encode(first_frame_image.to(weight_dtype)).latent_dist.mode()* vae.config.scaling_factor  

                # add depth pseudo image
                first_frame_depth_image = batch["pseudo_depth_values"].squeeze(0)
                first_frame_depth_image = first_frame_depth_image + noise_aug_strength * torch.randn_like(first_frame_depth_image)

                # don't understand this
                fps = 5
                motion_bucket_id = 127
                add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
                add_time_ids = torch.tensor(add_time_ids, device=latents.device).unsqueeze(0).expand(bsz, -1)
                # Predict the noise residual and compute loss

                model_pred = model(first_frame_image,
                                   first_frame_depth_image,
                                   encoder_hidden_states,
                                   noisy_latents,
                                   c_noise,
                                   add_time_ids,
                                   vae.config.scaling_factor,
                                   bsz,
                                   c_in,
                                   weight_dtype)

                pred_final = c_out*model_pred + c_skip*noisy_latents 

                loss = ((pred_final - latents)**2 * loss_weight).mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                # Grad accumulation
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    with FSDP.summon_full_params(model):
                        ema_unet.step(model.unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    layout_enc_ckpt = LayoutCondEncoder()
                    with FSDP.summon_full_params(model):
                        for (tname, tparam), (sname, sparam) in zip(list(layout_enc_ckpt.named_parameters()), list(model.layout_encoder.named_parameters())):
                            tparam.data.copy_(sparam.to(tparam.device).data)

                    layout_enc_depth_ckpt = LayoutCondEncoder()
                    with FSDP.summon_full_params(model):
                        for (tname, tparam), (sname, sparam) in zip(list(layout_enc_depth_ckpt.named_parameters()), list(model.layout_encoder_depth.named_parameters())):
                            tparam.data.copy_(sparam.to(tparam.device).data)

                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unet = UNetSpatioTemporalConditionModelV2_inchannel16.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                                                              in_channels=16,
                                                                                            out_channels=8,
                                                                                            low_cpu_mem_usage=False,
                                                                                            ignore_mismatched_sizes=True,
                                                                                              revision=args.non_ema_revision)
                        # unet = add_input_channels_conv(unet, 8)
                        if args.use_ema:
                            ema_unet.copy_to(unet.parameters())

                        pipeline = StableVideoDiffusionPipeline_convnb_mm(
                            image_encoder=image_encoder,
                            vae=vae,
                            unet=unet,
                            layout_encoder=layout_enc_ckpt,
                            layout_encoder_depth=layout_enc_depth_ckpt,
                            scheduler=noise_scheduler,
                            feature_extractor=feature_extractor,
                        )
                        pipeline.save_pretrained(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                pass

    layout_enc_ckpt = LayoutCondEncoder()
    with FSDP.summon_full_params(model):
        for (tname, tparam), (sname, sparam) in zip(list(layout_enc_ckpt.named_parameters()), list(model.layout_encoder.named_parameters())):
            tparam.data.copy_(sparam.to(tparam.device).data)

    layout_enc_depth_ckpt = LayoutCondEncoder()
    with FSDP.summon_full_params(model):
        for (tname, tparam), (sname, sparam) in zip(list(layout_enc_depth_ckpt.named_parameters()), list(model.layout_encoder_depth.named_parameters())):
            tparam.data.copy_(sparam.to(tparam.device).data)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = UNetSpatioTemporalConditionModelV2_inchannel16.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                                              in_channels=16,
                                                                              out_channels=8,
                                                                              low_cpu_mem_usage=False,
                                                                              ignore_mismatched_sizes=True,
                                                                              revision=args.non_ema_revision)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableVideoDiffusionPipeline_convnb_mm(
            image_encoder=image_encoder,
            vae=vae,
            unet=unet,
            layout_encoder=layout_enc_ckpt,
            layout_encoder_depth=layout_enc_depth_ckpt,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)

        # # Run a final round of inference.
        # images = []
        # if args.validation_prompts is not None:
        #     logger.info("Running inference for collecting generated images...")
        #     pipeline = pipeline.to(accelerator.device)
        #     pipeline.torch_dtype = weight_dtype
        #     pipeline.set_progress_bar_config(disable=True)

        #     if args.enable_xformers_memory_efficient_attention:
        #         pipeline.enable_xformers_memory_efficient_attention()

        #     if args.seed is None:
        #         generator = None
        #     else:
        #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        #     for i in range(len(args.validation_prompts)):
        #         with torch.autocast("cuda"):
        #             image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
        #         images.append(image)

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
