import os
import numpy as np
import cv2
import json

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



def load_config(config_path='motivation_config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def process_single_file(args,config_path='motivation_config.json'):
    config = load_config(config_path)
    yolov5_project_dir = config['yolov5_project_dir']
    faster_rcnn_project_dir = config['faster_rcnn_project_dir']
    confidence_threshold = config['confidence_threshold']
    area_threshold = config['area_threshold']
    gt_name, video_name, QP, res, segments, label = args
    tmp_data = np.zeros((segments, 5))
    src_name = video_name
    output_name = video_name
    video_totalname = f'{video_name}_{QP}_{res}_0'
    testpath = os.path.join(yolov5_project_dir, f"video_{video_name}_segments_qp_s", 'labels') #2
    if gt_name == "yolov5":
        stdpath = os.path.join(yolov5_project_dir, f"video_{video_name}_segments_x", 'labels')
    else:
        stdpath = os.path.join(faster_rcnn_project_dir, f"video_{video_name}_segments", 'labels')
    for count in range(1, segments + 1):
        testtxt = f'segment_{count}_qp{QP}_res{res}_frame'
        stdtxt = f'frame_{count}'
        tmp_data[count - 1, :] = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold = 0.5, confidence=confidence_threshold, area_threshold = area_threshold)
    # print(tmp_data)

    return tmp_data



def get_file_num(folder_path, prefix, suffix):
    """
    Count the number of files in a folder with a given prefix and suffix.
    """
    return sum(1 for filename in os.listdir(folder_path) if filename.startswith(prefix) and filename.endswith(suffix))

# for detection to get result
def compute_iou(rec1, rec2):
    """
    Calculate the intersection over union (IOU) of two rectangles.
    :param rec1: (xc, yc, w, h) representing the coordinates of the first rectangle.
    :param rec2: (xc, yc, w, h) representing the coordinates of the second rectangle.
    :return: The IOU (intersection over union) of the two rectangles.
    """
    ans1 = [(rec1[0] - rec1[2] / 2), (rec1[1] - rec1[3] / 2), (rec1[0] + rec1[2] / 2), (rec1[1] + rec1[3] / 2)]
    ans2 = [(rec2[0] - rec2[2] / 2), (rec2[1] - rec2[3] / 2), (rec2[0] + rec2[2] / 2), (rec2[1] + rec2[3] / 2)]

    left_column_max = max(ans1[0], ans2[0])
    right_column_min = min(ans1[2], ans2[2])
    up_row_max = max(ans1[1], ans2[1])
    down_row_min = min(ans1[3], ans2[3])
    # if the two rectangles have no overlapping region.
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    else:
        S1 = (ans1[2] - ans1[0]) * (ans1[3] - ans1[1])
        S2 = (ans2[2] - ans2[0]) * (ans2[3] - ans2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    return S_cross / (S1 + S2 - S_cross)

def check_file_status(filename):
    """
    Check if the file exists and is not empty.

    Args:
    - filename (str): The path to the file to check.

    Returns:
    - int: 0 if the file doesn't exist or is empty, 1 if the file exists and is not empty.
    """
    # Check if the file exists
    if not os.path.exists(filename):
        return 0
    # Check if the file is empty (file size is 0)
    elif os.path.getsize(filename) == 0:
        return 0
    else:
        return 1

def performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold, confidence=0.5, area_threshold = 0):
    confidence = float(confidence)
    area_threshold = float(area_threshold)
    if label == 0:
        label = [0]
    if label == 2:
        label = [2,7] # car and trunk
    if label == 1:
        label = [0,2,7] # complex
    path1 = os.path.join(stdpath, f'{stdtxt}.txt')
    path2 = os.path.join(testpath, f'{testtxt}.txt')

    # Check if the standard file exists and is not empty
    if not check_file_status(path1):
        print(f'{path1} doesn\'t exist or is empty')

        # Check if the test file exists and is not empty
        if not check_file_status(path2):
            return 0, 0, 0, 0, 0
        else:
            testfile = np.loadtxt(path2).reshape(-1, 6)
            # testfile = testfile[testfile[:, 5] >= confidence]
            testfile = testfile[np.isin(testfile[:, 0], label)]
            testfile = testfile[(testfile[:, 5] >= confidence) & (testfile[:, 3] * testfile[:, 4] > area_threshold)]
            return 0, 0, len(testfile), 0, 0

    # Check if the test file exists and is not empty
    elif not check_file_status(path2):
        print(f'{path2} doesn\'t exist or is empty')
        stdfile = np.loadtxt(path1).reshape(-1, 6)

        # stdfile = stdfile[stdfile[:, 0] == label]
        stdfile = stdfile[np.isin(stdfile[:, 0], label)]

        # stdfile = stdfile[stdfile[:, 5] >= confidence & stdfile[:, 3] * stdfile[:, 4] > area_threshold]
        stdfile = stdfile[(stdfile[:, 5] >= confidence) & ((stdfile[:, 3] * stdfile[:, 4]) > area_threshold)]


        FN = len(stdfile)
        return 0, FN, 0, 0, 0

    # Both files exist and are not empty
    else:
        # print("aa")
        stdfile = np.loadtxt(path1).reshape(-1, 6)

        # stdfile = stdfile[stdfile[:, 0] == label]
        stdfile = stdfile[np.isin(stdfile[:, 0], label)]

        # stdfile = stdfile[stdfile[:, 5] >= confidence & stdfile[:, 3] * stdfile[:, 4] > area_threshold]
        stdfile = stdfile[(stdfile[:, 5] >= confidence) & ((stdfile[:, 3] * stdfile[:, 4]) > area_threshold)]


        # stdfile = stdfile[(stdfile[:, 5] >= confidence) & (stdfile[:, 3] * stdfile[:, 4] > area_threshold)]

        testfile = np.loadtxt(path2).reshape(-1, 6)
        testfile = testfile[np.isin(testfile[:, 0], label)]
        testfile = testfile[(testfile[:, 5] >= confidence) & (testfile[:, 3] * testfile[:, 4] > area_threshold)]
        # testfile = testfile[testfile[:, 0] == label]
        # print(testfile.shape)


        TP, FN, iou_cum_recall, iou_cum_acc, = 0, 0, 0, 0
        matched = np.ones(len(testfile))  # Track which test boxes are available

        for line in stdfile:
            iou_list = [compute_iou(line[1:5], tline[1:5]) for tline in testfile]
            if iou_list:
                iou_list = np.array(iou_list)
                avi_iou_list = iou_list * matched
                result = np.max(avi_iou_list)
                max_index = np.argmax(avi_iou_list)
                iou_cum_recall += result
                if result >= threshold:
                    iou_cum_acc += result
                    TP += 1
                    matched[max_index] = 0
                else:
                    FN += 1
            else:
                FN += 1

        if TP != 0:
            return TP, FN, len(testfile) - TP, iou_cum_recall / (TP + FN), iou_cum_acc / TP
        elif FN != 0:
            return 0, FN, len(testfile), iou_cum_recall / FN, 0
        else:
            return TP, FN, len(testfile) - TP, 0, 0

def element2result(new_array):
    TP = new_array[..., :, 0]
    FN = new_array[..., :, 1]
    FP = new_array[..., :, 2]
    iou_cum_recall = new_array[..., :, 3]
    iou_cum_acc =  new_array[..., :, 4]

    epsilon = 1e-7
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    F1 = 2 * precision * recall / (precision + recall + epsilon)

    precision[(TP == 0) & (FP != 0)] = 0
    recall[(TP == 0) & (FN != 0)] = 0
    F1[TP == 0] = 0
    precision[(TP == 0) & (FP == 0)] = 1
    recall[(TP == 0) & (FN == 0)] = 1
    F1[(TP == 0) & (FP == 0) & (FN == 0)] = 1
    iou_cum_recall[(TP == 0) & (FN == 0)] = 1
    iou_cum_acc[(TP == 0)] = 1

    new_array[..., 0] = precision
    new_array[..., 1] = recall
    new_array[..., 2] = F1
    new_array[..., 3] = iou_cum_recall
    new_array[..., 4] = iou_cum_acc
    return new_array

