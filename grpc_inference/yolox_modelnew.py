from models.onnxnew  import multiclass_nms, demo_postprocess
from models.onnxnew import preprocess as preproc

import onnxruntime
import json
import logging
from configparser import ConfigParser
import numpy as np
import time
import cv2
from collections import namedtuple
class CombineModelYolox(object):
    def __init__(self, model_id, cfg_path):
        self.model_id = model_id
        self.cfg_path = cfg_path
        self.cfg_dic = self._config_parser()
        self.model_name = 'model_{}'.format(str(model_id))
        self.model_cfg = self.cfg_dic[self.model_name]
        self.model_detect_path=self.model_cfg['model_detect_path']
        self.conf_thres=self.model_cfg['conf_thres']
        self.iou_thres=self.model_cfg['iou_thres']
        self.class_mapper = json.loads(self.model_cfg['class_mapper'])
        self.thresh_values = [float(x) for x in self.model_cfg['thresh_values'].split(',')]
        self.num_classes = int(self.model_cfg['num_classes'])
        self.input_shape = self.model_cfg['input_shape']
        self.result = None

    def _config_parser(self):
        logging.info('config file loaded: {}'.format(self.cfg_path))
        conf = ConfigParser()
        conf.read(self.cfg_path)
        cfg_dict = {}
        for section in conf.sections():
            section_items = conf.items(section)
            section_dict = {}
            for pair in section_items:
                section_dict[pair[0]] = pair[1]
            cfg_dict[section] = section_dict
        return cfg_dict
    def detect(self, image):
        self.image_path = image
        input_shape = tuple(map(int, self.input_shape.split(',')))
        origin_img = self.image_path
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        #img, ratio = preproc(origin_img, input_shape, mean, std)
        img, ratio = preproc(origin_img, input_shape)
        session = onnxruntime.InferenceSession(self.model_detect_path)
        s_t1 = time.time()
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=False)[0]
        # self.logging.info('Start_time:{}'.format(time.time() - s_t))
        # self.logging.info('Inference_time:{}'.format(time.time() - s_t1))
        # self.logging.info('Get_device:{}'.format(onnxruntime.get_device()))
        # print('start_runtime:',time.time()-s_t)
        # print('inference_runtime:',time.time()-s_t1)
        # print('get_device:',onnxruntime.get_device())
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        pre = multiclass_nms(boxes_xyxy, scores, nms_thr=float(self.iou_thres), score_thr=0.1)
        if pre is not None:
            return pre
        else:
            return []

    def inference(self, image_np, photo_id, x_offset=0, y_offset=0):
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        pre= self.detect(image_np)
        self.result = self._mergeresult(pre)
    def _mergeresult(self, pre):
        total_boxes, total_labels, total_masks, new_points = [], [], [], []
        logging.info('pre_info_yolox{}'.format(pre))
        if len(pre)>0:
            for i in pre:
                x,y,x1,y1,s,c= i
                total_boxes.append(
                    [int(x), int(y), int(x1), int(y1), float(s)])
                total_labels.append(int(c))
                total_masks.append(np.array([]))
                new_points.append([])
        Result = namedtuple('Result', ['masks', 'bboxes', 'labels', 'points'])
        result = Result(masks=total_masks, bboxes=total_boxes, labels=total_labels, points=new_points)
        return result
