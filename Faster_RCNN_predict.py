# This code has been modified to save the data in yolov5 format and adapted to video processing.
# This faster rcnn implementation comes from come from https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
import glob
import os
import time
import json
import argparse
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



# YOLOv5 change1
def normalize_boxes(boxes, width, height):
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  

    
    normalized_xywh = np.zeros_like(xywh, dtype=float)
    normalized_xywh[:, 0] = xywh[:, 0] / width
    normalized_xywh[:, 1] = xywh[:, 1] / height
    normalized_xywh[:, 2] = xywh[:, 2] / width
    normalized_xywh[:, 3] = xywh[:, 3] / height

    return normalized_xywh



# YOLOv5 change2
def save_all_frames(video_folder, video_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if video_file.endswith(".mp4") or video_file.endswith(".avi"):  
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        true_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_name = f"{os.path.splitext(video_file)[0]}_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            count += 1
        cap.release()
    # cv2.destroyAllWindows()
    return count

# YOLOv5 change3
def save_txt_like_yolo(cls, xywh, conf, file, threshold):
    
    threshold = float(threshold)
    with open(file, 'w+') as f:
        for i in range(len(cls)):
            if conf[i]>threshold:
            # if (cls[i] == 1 or cls[i] == 3) and conf[i]>threshold:
                line = (cls[i]-1, *xywh[i], conf[i])
                f.write(('%g ' * len(line)).rstrip() % line + '\n')


def main():
    """
      YOLOv5 change3
      add 6 new variables
      images_dir,dataset_type,project,name,videofile,save_img
    """
    parser = argparse.ArgumentParser(description="FRCNN Demo.")


    parser.add_argument("--images_dir", default=str(ROOT)+ '/pictures', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument('--project', default=str(ROOT)+  '/runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument("--videofile", default='none', type=str, help='video')
    parser.add_argument('--save_img', action='store_true', help='show results')
    parser.add_argument("--output_dir", default='', type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument("--threshold", default='0.25', type=str,
                        help='confidence threshold')
    args = parser.parse_args()

    args.output_dir = f'{args.project}/{args.name}'


    if args.videofile != 'none':
        video_folder, video_file = os.path.split(args.videofile)
        # frames_folder = f'{video_folder}/{video_file[:-4]}'
        # save_all_frames(video_folder, video_file, frames_folder)
        # args.images_dir = frames_folder
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=91)

    # YOLOv5 change4
    weights_path = str(ROOT)+ "/save_weights/fasterrcnn_resnet50_fpn_coco.pth"
    print(weights_path)
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    # read class_indict
    label_json_path = str(ROOT) + '/coco91_indices.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)
    # YOLOv5 change5
    category_index = {str(v): str(k) for v, k in class_dict.items()}
    image_paths = glob.glob(os.path.join(args.images_dir, '*.jpg'))
    if not os.path.exists(args.project):
        os.mkdir(args.project)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    original_img = Image.open(str(ROOT) +'/pictures/1_short_0.jpg')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()  
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

    total_time = 0
    with torch.no_grad():
        # YOLOv5 change6
        if args.videofile == 'none':
            for i, image_path in enumerate(image_paths):

                t_start = time_synchronized()
                image_name = os.path.basename(image_path)
                original_img = Image.open(image_path)
                img_width, img_height = original_img.size
                # from pil image to tensor, do not normalize image
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)
                predictions = model(img.to(device))[0]
                t_end = time_synchronized()

                inference_time = t_end - t_start
                total_time += inference_time
                print("inference+NMS time: {}".format(t_end - t_start))

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                boxes = predict_boxes
                xywh = normalize_boxes(boxes, img_width, img_height)
                if not os.path.exists(f'{args.output_dir}/labels/'):
                    os.mkdir(f'{args.output_dir}/labels/')
                file = f'{args.output_dir}/labels/{image_name[:-4]}.txt'
                save_txt_like_yolo(predict_classes, xywh, predict_scores, file, args.threshold)

                if args.save_img:
                    plot_img = draw_objs(original_img,
                                         predict_boxes,
                                         predict_classes,
                                         predict_scores,
                                         category_index=category_index,
                                         box_thresh=0.5,
                                         line_thickness=3,
                                         font='arial.ttf',
                                         font_size=20)
                    # plt.imshow(plot_img)
                    # plt.show()
                    # Image.fromarray(plot_img).save(os.path.join(args.output_dir, image_name))
                    plot_img.save(os.path.join(args.output_dir, image_name))
            # YOLOv5 change7
            file_path = "time.txt"
            with open(file_path, "a+") as file_write:
                file_write.write(f"{total_time}, {total_time}, {args.images_dir}, faster_rcnn\n")
        # YOLOv5 change8
        else:
            if video_file.endswith(".mp4") or video_file.endswith(".avi"):  
                video_path = os.path.join(video_folder, video_file)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = 0
                true_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    t_start = time_synchronized()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image
                    original_img = Image.fromarray(np.uint8(frame))
                    img_width, img_height = original_img.size

                    # from pil image to tensor, do not normalize image
                    data_transform = transforms.Compose([transforms.ToTensor()])
                    img = data_transform(original_img)
                    # expand batch dimension
                    img = torch.unsqueeze(img, dim=0)

                    predictions = model(img.to(device))[0]
                    t_end = time_synchronized()

                    inference_time = t_end - t_start
                    total_time+=inference_time
                    print("inference+NMS time: {}".format(inference_time))

                    predict_boxes = predictions["boxes"].to("cpu").numpy()
                    predict_classes = predictions["labels"].to("cpu").numpy()
                    predict_scores = predictions["scores"].to("cpu").numpy()
                    boxes = predict_boxes
                    # if len(predict_boxes) == 0:
                    xywh = normalize_boxes(boxes, img_width, img_height)
                    if not os.path.exists(f'{args.output_dir}/labels/'):
                        os.mkdir(f'{args.output_dir}/labels/')
                    file = f'{args.output_dir}/labels/{video_file[:-4]}_{count}.txt'
                    save_txt_like_yolo(predict_classes, xywh, predict_scores, file,args.threshold)
                    if args.save_img:
                        # if not os.path.exists(args.output_dir):
                        #     os.mkdir(args.output_dir)
                        plot_img = draw_objs(original_img,
                                             predict_boxes,
                                             predict_classes,
                                             predict_scores,
                                             category_index=category_index,
                                             box_thresh=0.5,
                                             line_thickness=3,
                                             font='arial.ttf',
                                             font_size=20)
                        plt.imshow(plot_img)
                        plt.show()
                        # Image.fromarray(plot_img).save(os.path.join(args.output_dir, f'{video_file[:-4]}_{count}'))
                        plot_img.save(os.path.join(args.output_dir, f'{video_file[:-4]}_{count}'))
                    count+= 1
                cap.release()
            file_path = "time.txt"
            with open(file_path, "a+") as file_write:
                file_write.write(f"{total_time}, {video_file}, faster_rcnn\n")
if __name__ == '__main__':
    main()
    ## python D:/code/PycharmProjects/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/predict.py  --project D:/code/PycharmProjects/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/runs/  --name 003 --videofile D:/source/video_1.mp4
