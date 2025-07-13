import os
import shutil

def RemoveDir(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

def main(mvs_input_path, mvs_output_path,clear_cam_list):
    RemoveDir(mvs_input_path)
    for cam in clear_cam_list:
        rm_output_dir = os.path.join(mvs_output_path,cam)
        RemoveDir(rm_output_dir)
