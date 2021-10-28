#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import numpy as np
import json
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


def instance_to_json(instance, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        content = json.dumps(instance, ensure_ascii=False, indent=2)
        f.write(content)


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".JPG"]

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default='yolox-s', help="model name")

    parser.add_argument(
        "--path", default="/home/jerry/data/Micro_A/A_loushi/combined/cm_daowen/cm_ps_dw", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=True, 
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='/home/jerry/Documents/YOLOX/YOLOX_outputs/10-21-cm-ps-dw/2021102118:56:01/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    # parser.add_argument("--tsize", nargs='+', default=None, type=int, help="test img size")

    parser.add_argument('--tsize', nargs='+', type=int, default=None, help='train,test sizes')
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))    
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
       
        return vis_res


def write_json(save_file_name, results, img_info):

    results = results[0]

    results = np.array(results.cpu(), np.float64)
    img_name = img_info['file_name']
    ratio = img_info['ratio']

    shapes = []
    num_detections = len(results)
    for result in results:
        shape = {}
        score = round(result[4] * result[5], 4)
        class_name = COCO_CLASSES[int(result[6])]

        x1, y1, x2, y2 = result[0:4] / ratio

        x1 = int(min(x1, x2))
        x2 = int(max(x1, x2))
        y1 = int(min(y1, y2))
        y2 = int(max(y1, y2))

        bw = x2-x1
        bh = y2-y1

        shape['label'] = class_name
        shape['shape_type'] = 'rectangle'
        shape['points'] = [[x1, y1], [x2, y2]]
        shape['status'] = '0'
        shape['describe'] = 'inference'

        shapes.append(shape)
    
    instance = {'version': '1.0',
                'shapes': shapes,
                'imageData': None,
                'imageWidth': img_info['width'],
                'imageHeight': img_info['height'],
                'imageDepth': 3,
                'imagePath': img_name
                }
    instance_to_json(instance, save_file_name[:save_file_name.rindex('.')]+'.json')

def write_txt(save_file_name, results, img_info, img_shape):
    results = results[0]

    results = np.array(results.cpu(), np.float64)
    img_name = img_info['file_name']
    ratio = img_info['ratio']


    with open(save_file_name[:save_file_name.rindex('.')]+'.txt', 'w') as f:
        width, height = img_shape[1], img_shape[0]
        for result in results:
            score = round(result[4] * result[5], 4)
            class_name = COCO_CLASSES[int(result[6])]
            x1, y1, x2, y2 = result[0:4] / ratio

            x1 = int(min(x1, x2))
            x2 = int(max(x1, x2))
            y1 = int(min(y1, y2))
            y2 = int(max(y1, y2))

            bw = x2-x1
            bh = y2-y1
            try:
                w, h = bw, bh
                cx, cy = x1+bw/2, y1+bh/2
                box_info = "%d %.03f %.03f %.03f %.03f %.03f" % (int(result[6]), cx/width, cy/height, w/width, h/height, score)
                f.write(box_info)
                f.write('\n')
            except:
                print('{}异常，请检查'.format(save_file_name))

def write_csv(results, img_info):

    import csv
    results = results[0]
    # print(results)
    results = numpy.array(results.cpu(), numpy.float64)
    img_name = img_info['file_name']

    img = cv2.imread(os.path.join('/home/jerry/data/Micro_A/A_loushi/combined/cm_daowen/cm_ps_dw', img_name))
    # print()
    header = ['photo_id', 
              'product_id', 
              'channel_id', 
              'class_name', 
              'num_detections', 
              'xmin', 
              'ymin', 
              'bb_width', 
              'bb_height', 
              'score', 
              'length', 
              'width', 
              'pixel_area', 
              'gradients', 
              'contrast', 
              'brightness', 
              'min_distance']
    data_list = []
    num_detections = len(results)
    for result in results:
        score = round(result[4] * result[5], 4)
        class_name = COCO_CLASSES[int(result[6])]
        x1, y1, x2, y2 = result[0:4]

        x1 = int(min(x1, x2))
        x2 = int(max(x1, x2))

        y1 = int(min(y1, y2))
        y2 = int(max(y1, y2))
        bw = x2-x1
        bh = y2-y1

        data = ['1','1','1', class_name, num_detections, str(x1), str(y1), (str(bw)), str(bh), str(score), '1', '1', '1', '1', '1', '1']

        data_list.append(data)

    # import matplotlib.pyplot as plt 
    # img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 10)
    # plt.imshow(img)
    # plt.show()
    csv_name = img_name.replace('.jpg', '.csv')
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, csv_name), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_list)

    # pass


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

        if outputs[0] == None:
            continue

        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

            # write_json(save_file_name, outputs, img_info)
            # write_txt(save_file_name, outputs, img_info, result_image.shape)
            # write_csv(outputs, img_info)
            
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    from modify_args import modify_args
    exp = modify_args(exp)
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
