<!-- # <center> DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation -->

<div align="center">

___<font size="10">***DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation***</font>___
<br>
<br>
<a href='https://arxiv.org/pdf/2503.15208'><img src='https://img.shields.io/badge/arXiv-2503.15208-b31b1b.svg'></a>
&nbsp;
<a href='https://royalmelon0505.github.io/DiST-4D/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
&nbsp;

_**[Jiazhe Guo](https://scholar.google.com/citations?hl=zh-CN&user=pWLqZnoAAAAJ), [Yikang Ding](https://scholar.google.com/citations?hl=zh-CN&user=gdP9StQAAAAJ), [Xiwu Chen](https://scholar.google.com/citations?user=PVMQa-IAAAAJ), [Shuo Chen](), [Bohan Li](https://scholar.google.com/citations?hl=zh-CN&user=V-YdQiAAAAAJ), [Yingshuang Zou](https://heiheishuang.xyz), <br>[Xiaoyang Lyu](https://shawlyu.github.io/), [Feiyang Tan](https://scholar.google.com/citations?hl=zh-CN&user=KeiZBdMAAAAJ), [Xiaojuan Qi](https://scholar.google.com/citations?hl=zh-CN&user=bGn0uacAAAAJ), [Zhiheng Li](https://www.sigs.tsinghua.edu.cn/lzh_en/main.htm), [Hao Zhao](https://scholar.google.com/citations?hl=zh-CN&user=ygQznUQAAAAJ)**_
<br><br>
<font size="10">ICCV 2025</font>
</div>

<!-- ## DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation -->

<!-- [![arXiv paper](https://img.shields.io/badge/arXiv%20%2B%20supp-2503.-purple)](https://arxiv.org/abs/)
[![Code page](https://img.shields.io/badge/Project%20Page-DiST4D-red)](https://royalmelon0505.github.io/DiST-4D/) -->


*DiST-4D* is the first framework to achieve **feed-forward** dynamic 4D driving scene generation with both temporal extrapolation and spatial novel view synthesis.

## 💫  Framework
<div align=center><img width="960" height="372" src="./assets/Fig_ppl.png"/></div>

DiST-4D is a disentangled spatiotemporal diffusion framework for 4D driving scene generation, leveraging metric depth as the core geometric representation to enable both temporal extrapolation and spatial novel view synthesis (NVS). 
**(Top: Temporal Generation)** DiST-T employs a diffusion model to predict future multi-camera RGB-D sequences from historical multi-camera images and control signals. The generated RGB-D sequences are then aggregated into point clouds, allowing for bullet time rendering. **(Bottom: Spatial Generation)** To enable spatial NVS, DiST-S leverages the predicted RGB-D sequences to generate novel viewpoints by first projecting them into sparse conditions and then refining them into dense RGB-D outputs.

## 🔆 News
- [2025/7]: Code and pre-trained weights are released.
- [2025/6]: Paper is accepted on **ICCV 2025**.
- [2025/3]: Check out our other latest works on generative world models: [UniScene](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/blob/master/README.md), [MuDG](https://github.com/heiheishuang/MuDG), [HERMES](https://lmd0311.github.io/HERMES/).
- [2025/3]: Paper is on [arxiv](https://arxiv.org/abs/2503.15208).
- [2025/3]: Demo is released on [Project Page](https://royalmelon0505.github.io/DiST-4D/).

## 👀 Abstract
Current generative models struggle to synthesize dynamic 4D driving scenes that simultaneously support temporal extrapolation and spatial novel view synthesis (NVS) without per-scene optimization. A key challenge lies in finding an efficient and generalizable geometric representation that seamlessly connects temporal and spatial synthesis. To address this, we propose DiST-4D, the first disentangled spatiotemporal diffusion framework for 4D driving scene generation, which leverages metric depth as the core geometric representation. DiST-4D decomposes the problem into two diffusion processes: DiST-T, which predicts future metric depth and multi-view RGB sequences directly from past observations, and DiST-S, which enables spatial NVS by training only on existing viewpoints while enforcing cycle consistency. This cycle consistency mechanism introduces a forward-backward rendering constraint, reducing the generalization gap between observed and unseen viewpoints. Metric depth is essential for both accurate reliable forecasting and accurate spatial NVS, as it provides a view-consistent geometric representation that generalizes well to unseen perspectives. Experiments demonstrate that DiST-4D achieves state-of-the-art performance in both temporal prediction and NVS tasks, while also delivering competitive performance in planning-related evaluations.


## 🕹️ Getting Started

### Prepare NuScenes Dataset
1. Download nuScenes dataset 
Download all splits of Trainval in Full dataset (v1.0) to your device following nuscenes official instructions and put them to "/data/nuscenes".

2. Download advanced_12Hz_trainval metadata from [MagicDrive](https://coda-dataset.github.io/w-coda2024/track2/)
```
$./data/nuscenes
├── samples
├── sweeps
├── ...
├── v1.0-trainval
└── advanced_12Hz_trainval
```

3. Download metadata (preprocessing pkl) as [MagicDriveDiT](https://github.com/flymin/MagicDriveDiT) and put them to "/data/nuscenes_mmdet3d-12Hz/"

### Preprocess Depth of NuScenes

1. conda environment

- pre_py39: `depth_process/requirement/pre_py39.txt` which is modified from [DepthLab](https://github.com/ant-research/DepthLab)

- mvs_py38: `depth_process/requirement/mvs_py38.txt` which is used for semantic segmentation in visual reconstruction.

2. DownLoad the ckpt of [DepthLab](https://github.com/ant-research/DepthLab) in `depth_process/DepthLab/checkpoints`

3. Prepare the ckpt of [SegFormer](https://github.com/NVlabs/SegFormer) to `depth_process/visualrecon/SemSeg/SegFormer` and [Oneformer](https://github.com/SHI-Labs/OneFormer) to `depth_process/visualrecon/SemSeg/OneFormer`

4. Preprocess one scene in NuScenes (for example indice = 1)
```bash
# MVS part
cd depth_process
./visualrecon/mvs_pipe_nus.sh $indice

# DepthLab refine depth
./DepthLab/scripts/infer_nus_video_mp.sh $indice
```
**Note:**  In order to preprocessing all scene, you need to run the above script from `indice=0` to `indice=849`, which is time consuming. Since we already have multi-processing setup in our MVS code, it is more recommended to use a loop to process each scene like `depth_process/visualrecon/batch_mvs_nus.sh`

5. Semantic Segmenation with [Oneformer](https://github.com/SHI-Labs/OneFormer) form scene-0 to scene-849 using 8gpu and mp=1
```bash
cd depth_process/visualrecon
./Batch_SemSeg_nus_oneformer.sh 0 849 8 1

# We recommend to use multiple processes or nodes to handle the above tasks
#./Batch_SemSeg_nus_oneformer.sh 0 99 8 1
#./Batch_SemSeg_nus_oneformer.sh 100 199 8 1
# ...
```

The processed depth and semantic will be put in `depth_process/nus_Rdepth` (A scene takes up about 3GB)

We have also provided some preprocessed scenes on the validation set to facilitate testing. Please download these scenes from [Hugging Face](https://huggingface.co/datasets/melon287/nus_Rdepth/blob/main/release_data_nus_Rdepth.zip)

### DiST-T
1. conda environment: 

- distt: `DiST_T/requirement/distt.txt` which is modified from [MagicDriveDiT](https://github.com/flymin/MagicDriveDiT)

2. Prepare CogVideoX VAE and T5 emcoder following [MagicDriveDiT](https://github.com/flymin/MagicDriveDiT) and put them in `depth_process/DepthLab/checkpoints`

3. Training and inference code. 
```bash
# Train 
cd DiST_T
./train_dist_mm_424_onlyRGB.sh

# Infer
cd DiST_T
./infer_dist_dataset_fullval.sh
```
#### Only inference the DiST-T model for test 

1. Download the pretrained [DiST-T ckpt](https://drive.google.com/drive/folders/1J5xI1RCT5MtEQvxsh76bs7H4wWsOW-dH?usp=drive_link) and put it in `DiST_T/ckpt/outputs_424_onlyRGB/`
2. Run the inference code
```bash
./infer_dist_dataset_fullval.sh
```

### DiST-S
1. conda environment
- dists: `DiST_S/requirement/dists.txt` which is modified from [FreeVS](https://github.com/esdolo/FreeVS)

2. Data process
```bash
cd DiST_S
#Train set forward or backward projection (current only +2frame)
./data_process/batch_reproj_nus_train_2frame.sh $start_idx $end_idx
#For example:
#./data_process/batch_reproj_nus_train_2frame.sh 0 700

#Val set lateral camera move (+1m +2m +4m)
./data_process/batch_reproj_nus_val_movecam.sh $start_idx $end_idx
#For example:
#./data_process/batch_reproj_nus_val_movecam.sh 0 150
```
**Note:**  We recommend to use multiple processes for data process. The processed condition is in `DiST_S/reproj_nus`

3. Download SVD ckpt from  https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt. and put it in `DiST_S/pretrained`

4. Train stage1
```bash
cd DiST_S/diffusers/
./examples/scripts/mm_train_nus_768.sh
```

5. SCC Data process
```bash
cd DiST_S/diffusers/
./examples/scripts/mm_infer_nus_cycle_reproj.sh
```

6. Train stage2 with SCC Data
```bash
cd DiST_S/diffusers/
./examples/scripts/mm_train_nus_modVis.sh
```

7. Final inference after SCC training
```bash
cd DiST_S/diffusers/
./examples/scripts/mm_infer_nus_mp_768_modVis.sh
```

#### Only inference the DiST-S model for test 

1. Run the data process of Val set. 
```bash
# lateral camera move (+1m +2m +4m) 
./data_process/batch_reproj_nus_val_movecam.sh $start_idx $end_idx
```

2. Download the pretrained [DiST-S ckpt](https://drive.google.com/drive/folders/1rKHV6to2J_frU7WOkfLergJjik02HN46?usp=sharing) and put it in `DiST_S/ckpt/mm_multi_cam_hybridD_valid_mask_768_mod_cycle_p`

3. Run the inference code
```bash
cd DiST_S/diffusers/
./examples/scripts/mm_infer_nus_mp_768_modVis.sh
```



## 🙏 Acknowledgements
We would like to thank the contributors of the following repositories for their valuable contributions to the community:
- [TransMVSNet](https://github.com/megvii-research/TransMVSNet)
- [DepthLab](https://github.com/ant-research/DepthLab)
- [MagicDriveDiT](https://github.com/flymin/MagicDriveDiT)
- [FreeVS](https://github.com/esdolo/FreeVS)
- [ViewCrafter](https://github.com/Drexubery/ViewCrafter)




## 😉 Citation
If you find our paper and code useful for your research, please consider citing:

```bibtex
@article{guo2024dist4d,
  title={DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation},
  author={Jiazhe Guo and Yikang Ding and Xiwu Chen and Shuo Chen and Bohan Li and Yingshuang Zou and Xiaoyang Lyu and Feiyang Tan and Xiaojuan Qi and Zhiheng Li and Hao Zhao},
  journal={arXiv preprint arXiv:2503.15208},
  year={2025}
}
```
