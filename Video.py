import os
import numpy as np
import subprocess
import cv2
import time
import matplotlib.pyplot as plt
import json

def load_config(config_path='motivation_config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def ensure_dir(directory):
    """
    Ensure the given directory exists, create it if it does not.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_yolo_imgs(source_dir, video_names, version, config_path='motivation_config.json'):
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    weights_dir = config['weights_dir']
    yolov5_ORGCA_project_dir = config['yolov5_ORGCA_project_dir']
    ensure_dir(yolov5_ORGCA_project_dir)
    for video_name in video_names:
        subprocess.run(
            [
                "python",
                yolov5_script_path,
                "--weights",
                os.path.join(weights_dir, f"yolov5{version}.pt"),
                "--project",
                yolov5_ORGCA_project_dir,
                "--name",
                f"{video_name}_{version}",
                "--source",
                os.path.join(source_dir, video_name),
                "--nosave",
                "--save-txt",
                "--save-conf",
            ]
        )
        # break

def run_yolo_imgs_one(source_dir, video_name, version, config_path='motivation_config.json'):
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    weights_dir = config['weights_dir']
    yolov5_ORGCA_project_dir = config['yolov5_ORGCA_project_dir']
    ensure_dir(yolov5_ORGCA_project_dir)
    subprocess.run(
        [
            "python",
            yolov5_script_path,
            "--weights",
            os.path.join(weights_dir, f"yolov5{version}.pt"),
            "--project",
            yolov5_ORGCA_project_dir,
            "--name",
            f"{video_name}_{version}",
            "--source",
            source_dir,
            "--nosave",
            "--save-txt",
            "--save-conf",
        ]
    )

def faster_rcnn_images(source_dir, video_names, config_path='motivation_config.json'):
    config = load_config(config_path)
    faster_rcnn_script_path = config['faster_rcnn_script_path']
    faster_rcnn_project_dir = config['faster_rcnn_project_dir']
    confidence_threshold = config['confidence_threshold']
    ensure_dir(faster_rcnn_project_dir)
    for video_name in video_names:
        subprocess.run(
            [
                "python",
                faster_rcnn_script_path,
                "--project",
                faster_rcnn_project_dir,
                "--name",
                f"{video_name}",
                "--images_dir",
                os.path.join(source_dir, video_name),
                "--threshold",
                confidence_threshold,
            ]
        )

def get_txt_path(video_name, dnn, version, config_path='motivation_config.json'):
    config = load_config(config_path)

    project_dirs = {
        'yolov5': config['yolov5_project_dir'],
        'faster_rcnn': config['faster_rcnn_project_dir'],
    }

    if dnn == 'yolov5' and version is not None:
        txt_path = os.path.join(project_dirs[dnn], f"{video_name}_{version}", "labels")
    else:
        txt_path = os.path.join(project_dirs[dnn], video_name, "labels")
    return txt_path

def yolov5(yolo_detect, yolo_weight, yolo_path, source_dir, video_name):
    subprocess.run(
        [
            "python",
            f"{yolo_detect}",
            "--weights",
            f"{yolo_weight}",
            "--project",
            f"{yolo_path}",
            "--name",
            f"{video_name}",
            "--source",
            f"{source_dir}{video_name}.mp4",
            "--nosave",
            "--save-txt",
            "--save-conf",
        ]
    )
