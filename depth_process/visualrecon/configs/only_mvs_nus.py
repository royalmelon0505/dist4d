indice = 3
split_num = 3
# device config
device_cfg = dict(
    gpu_num = 2,
    colmap_exe = f"SfM/colmap_3.8/build/src/exe/colmap"
)

# action_list
# full = ["data_process", "sfm", "sfm2mvs", "semseg", 
# "mvs_depth", "depth_fuse", "merge_pcd", "format_labeling"]
action_list = [
    # "data_process",
    # "sfm",
    # "sfm2mvs",
    # "semseg",
    "mvs_depth",
    "depth_fuse",
    "merge_pcd",
    # "format_labeling"
    "clear_MVS_mid"
]


semseg_cfg = dict(
    model = 'segformer', # ['segformer', 'oneformer', 'RV']
    segformer_cfg = dict(
        input_root = f"/data/code/visualrecon/MVS_Scene{indice}_input",
        save_root_ = f"/data/code/visualrecon/MVS_Scene{indice}_input/SemSeg",
        scale = 0.25,
        gpu_num = device_cfg["gpu_num"],
        mp_num= 4,
        palette = 'cityscapes',
        config = "SemSeg/SegFormer/local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py",
        checkpoint = "SemSeg/SegFormer/checkpoints/segformer.b3.1024x1024.city.160k.pth",
    ),
    oneformer_cfg = dict(
        input_path = f"/data/code/visualrecon/MVS_Scene{indice}_input",
        output = f"/data/code/visualrecon/MVS_Scene{indice}_input/SemSeg",
        task = 'semantic',
        scale = 0.25,
        gpu_num = device_cfg["gpu_num"],
        mp_num = 1,
        config_file = "SemSeg/OneFormer/configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml",
        opts = ["MODEL.IS_TRAIN", False, "MODEL.IS_DEMO", True, "MODEL.WEIGHTS", "SemSeg/OneFormer/ckpt/250_16_dinat_l_oneformer_mapillary_300k.pth"]
    ),
)

mvs_base_cfg = dict(
    testlist = "/data/code/visualrecon/MVS/TransMVSNet_hr/lists/nus_li.txt",
    ckpt = f"MVS/TransMVSNet_hr/ckpt/model_bld_hr.ckpt",
    mvs_in = f"/data/code/visualrecon/MVS_Scene{indice}_input",
    mvs_out = f"/data/code/visualrecon/MVS_Scene{indice}_output"
)

mvs_depth_cfg = dict(
    dataset = "nus_recon_dataset",
    num_view = 7,
    batch_size=device_cfg['gpu_num']*4,
    interval_scale = 1.0,
    numdepth = 192,
    ndepths = [48, 32, 8],
    depth_inter_r = [4, 1, 0.5],
    cr_base_chs = [8, 8, 8],
    max_h = 900, # 1200,    # 如果调大reso, bs可能需要减小
    max_w = 1600, # 2000,
    fix_res = False,
    num_worker = 4,
    save_vis = True,
    save_rgb = False,
    testpath = mvs_base_cfg["mvs_in"],
    testlist = mvs_base_cfg["testlist"],
    outdir = mvs_base_cfg["mvs_out"],
    loadckpt = mvs_base_cfg["ckpt"],
)

depth_fuse_cfg = dict(
    testpath=mvs_base_cfg['mvs_out'],
    input_path = mvs_base_cfg["mvs_in"],
    testlist = mvs_base_cfg['testlist'],
    outdir = mvs_base_cfg["mvs_out"],
    mp_num = 16,
    scale = 0.8,
    photo_threshold = 0.1,#0.05,
    thres_view = 3,#3, # 4
    Semantic_dilate_size = 0,
    Semantic_path = f"/data/code/visualrecon/MVS_Scene{indice}_SemSeg",
    save_sem_color = False,
    save_vis = True
)

merge_pcd_cfg = dict(
    src_path = mvs_base_cfg["mvs_out"],
    save_path = f"{mvs_base_cfg['mvs_out']}/sp_pcd",
    cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],#['cam_front_120', 'cam_back_120'],
    ds = True,
    ds_cls = 'voxel',
    voxel_size = 0.1,
    split_num = split_num,
    fuse_sem_color = False
)


clear_cache_cfg = dict(
    mvs_input_path = mvs_base_cfg["mvs_in"],
    mvs_output_path =  mvs_base_cfg["mvs_out"],
    clear_cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],#['cam_front_120', 'cam_back_120'],
)