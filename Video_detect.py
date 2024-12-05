from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np
from Video import *
from Performance import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_data_experiment(config_path='config.json'):
    config = load_config(config_path)
    video_names = [f"video_{i}_segments" for i in range(1,44)]
    source_dir = config["experiment_source_dir"]
    faster_rcnn_images(source_dir, video_names, config_path='config.json')
    # run_yolo_imgs(source_dir, video_names, "s" ,config_path='config.json')
    run_yolo_imgs(source_dir, video_names, "x" ,config_path='config.json')
    for video_name in video_names:
        video_dir = os.path.join(source_dir, video_name, "qp_encode")
        run_yolo_imgs_one(video_dir, video_name + "_qp", "s" ,config_path='config.json')

def save_data_experiment(des_name, config_path='config.json'):
    config = load_config(config_path)
    data_des_path = config['experiment_data_des_path']
    ensure_dir(data_des_path)
    gts = config['gts']
    QP_list = [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]  # QP
    res_list = [1, 0.7, 0.5, 0.3]  # 
    video_names = [i for i in range(1,44)]
    num_counts = 300
    label = int(config['car_label'])
    figure1_data = np.zeros((len(gts), len(video_names), len(QP_list), len(res_list), num_counts, 5))
    tasks = []
    for gt, gt_name in enumerate(gts):
        for video_index, video_name in enumerate(video_names):
           for QP_index, QP in enumerate(QP_list):
                for res_index, res in enumerate(res_list):
                    tasks.append((gt_name, video_name, QP, res, num_counts, label))
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_file, tasks))
    for i, result in enumerate(results):
        gt_name, video_name, QP, res, num_counts, label = tasks[i]
        QP_index = QP_list.index(QP)
        res_index = res_list.index(res)
        gt_index = gts.index(gt_name)
        video_index = video_names.index(video_name)
        figure1_data[gt_index, video_index, QP_index, res_index, :, :] = result
    np.save(os.path.join(data_des_path, f"{des_name}.npy"), figure1_data)

if __name__ == '__main__':
    run_data_experiment()
    des_name = f"experiment_knobs_segment_acc"
    save_data_experiment(des_name)
