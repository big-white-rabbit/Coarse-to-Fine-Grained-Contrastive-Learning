import os
import pickle
import pandas as pd
from detectron2.structures import BoxMode
import argparse
import dataclasses
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path
from typing import Any, Union, Dict, List
from math import ceil
from numpy import ndarray
import math
import h5py
from os.path import exists
import pydicom
import csv
import yaml
from PIL import Image

import cv2
import detectron2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict
import matplotlib.pyplot as plt

from detectron2.config.config import CfgNode as CN
from get_bbox_id import inference
from utils import get_kg2

from train_mimic_disease import Model
from torchvision import transforms

setup_logger()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

bboxlist = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone',
            'right hilar structures', 'right apical zone', 'right costophrenic angle',
            'right hemidiaphragm', 'left lung', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone',
            'left hilar structures', 'left apical zone', 'left costophrenic angle', 'left hemidiaphragm',
            'spine', 'aortic arch', 'mediastinum', 'upper mediastinum',
            'cardiac silhouette', 'left cardiac silhouette', 'right cardiac silhouette', 'cavoatrial junction'
]

diseaselist = ['lung opacity','pleural effusion','atelectasis','lobar/segmental collapse','enlarged cardiac silhouette','airspace opacity',
                'consolidation','tortuous aorta','calcified nodule','lung lesion','mass/nodule (not otherwise specified)',
               'pulmonary edema/hazy opacity','costophrenic angle blunting','vascular congestion','vascular calcification',
               'linear/patchy atelectasis','elevated hemidiaphragm','pleural/parenchymal scarring','hydropneumothorax','pneumothorax',
               'enlarged hilum','multiple masses/nodules','hyperaeration','mediastinal widening','vascular redistribution',
               'mediastinal displacement','spinal degenerative changes','superior mediastinal mass/enlargement','scoliosis',
               'sub-diaphragmatic air','hernia','spinal fracture','bone lesion','increased reticular markings/ild pattern',
               'infiltration','pneumomediastinum','bronchiectasis','cyst/bullae']

bbox_dict = {}
for i, bbox_class in enumerate(bboxlist):
    bbox_dict[bbox_class] = i

disease_dict = {}
for i, disease_class in enumerate(diseaselist):
    disease_dict[disease_class] = i

# --- setup ---

def format_pred(labels: ndarray, boxes: ndarray, scores: ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)



def predict_batch(predictor: DefaultPredictor, im_list: List[ndarray]) -> List:
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to('cuda')
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictor.model.to('cuda')
        predictions = predictor.model(inputs_list)
    return predictions


def get_mimic_ana_dicts(dataset_dir = 'chest-imagenome/1.0.0/silver_dataset/scene_graph'): # ordered csv. chest-imagenome is required. https://physionet.org/content/chest-imagenome/1.0.0/
    if exists('dictionary/mimic_ana_dicts.pkl'):
        with open('dictionary/mimic_ana_dicts.pkl', "rb") as tf:
            data_dicts = pickle.load(tf)
        return data_dicts
    data_dicts = []
    json_files = sorted(os.listdir(dataset_dir))
    category_set = {}

    n = 0
    for idx in tqdm(range(len(json_files))):
        json_file = json_files[idx]
    # for idx, json_file in enumerate(json_files):
        record = {}
        objs = []
        path = os.path.join(dataset_dir, json_file)
        with open(path) as f:
            data = json.load(f)
            record['file_name'] = '/mnt/data/wy/dataset/mimic_cxr_png/' + data['image_id'] + '.png'
            record['image_id'] = idx
            record['height'] = 1024
            record['width'] = 1024
            for object in data['objects']:
                try:
                    ratio = object['width']/object['original_width']
                except:
                    n += 1
                x1 = object['original_x1']
                y1 = object['original_y1']
                x2 = object['original_x2']
                y2 = object['original_y2']
                x1 = x1 * ratio / float(224/1024)
                x2 = x2 * ratio / float(224/1024)
                y1 = y1 * ratio / float(224/1024)
                y2 = y2 * ratio / float(224/1024)
                if object['name'] not in category_set:
                    category_set[object['name']] = len(category_set)
                obj = {
                    'bbox': [x1,y1,x2,y2],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    "category_id": category_set[object['name']]
                }
                objs.append(obj)
            record['annotations'] = objs
        data_dicts.append(record)
    print('total skipped', n)
    with open("dictionary/category_ana.pkl", "wb") as tf:
        pickle.dump(category_set, tf)
        print('dicts saved')
    with open("dictionary/mimic_ana_dicts.pkl", "wb") as tf:
        pickle.dump(data_dicts, tf)
        print('dicts saved')
    return data_dicts


def get_mimic_dict(full=True):
    datadir = '/mnt/data/wy/dataset/mimic_cxr_png/'
    datadict = []
    if full:
        name = '/mnt/data/wy/PythonProject/EKAID/data/mimic_shape_full.pkl'
    else:
        name = '/home/xinyue/VQA_ReGat/data/mimic/mimic_shape.pkl'
    with open(name, 'rb') as f:
        mimic_dataset = pickle.load(f)
    for i,row in enumerate(mimic_dataset):
        record = {}
        filename = datadir + row['image'] + '.png'
        record["file_name"] = filename
        record["image_id"] = row['image']
        record["height"] = 1024
        record["width"] = 1024
        datadict.append(record)

    return datadict

def save_features(mod, inp, outp):
    feats = inp[0]
    for i in range(batch_size):
        features.append(feats[i*1000: (i+1)*1000])
    predictions.append(outp)
    # feature = inp[0]
    # prediction = outp

def save_features1(mod, inp, outp):
    proposals.append(inp[2])
def get_img_list(img, bboxes):
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    totensor = transforms.ToTensor()
    img_list = []
    for bbox in bboxes:
        if ((int(bbox[2])-int(bbox[0])) * (int(bbox[3])-int(bbox[1])) != 0):
            new_img = totensor(img)
            new_img = new_img[:, int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
            new_img = resize(new_img)
            new_img = normalize(new_img)
        else:
            new_img = torch.zeros((3, 224, 224))
        img_list.append(new_img)
    return torch.stack(img_list)

def predict_disease(model, img, bbox, ann_class):
    model.eval()
    img_list = get_img_list(img, bbox)
    bbox, img_list, ann_class = bbox.to('cuda'), img_list.to('cuda'), ann_class.int().to('cuda')
    with torch.no_grad():
        feats, pred, binary_pred = model._forward(img_list, ann_class)

    _, binary_pred = torch.max(binary_pred, 1)
    pred = pred.ge(0.5)
    matrix = get_matrix(binary_pred, pred, ann_class)

    return feats, pred.int(), matrix

def get_matrix(binary_pred, pred, ann_class):
    bbox_length = len(bboxlist)
    disease_length = len(diseaselist)
    total_length = bbox_length + disease_length

    adj_matrix = np.zeros([100, 100], int)
    for i, ann in enumerate(ann_class):
        if binary_pred[i] != 0:
            for j, disease in enumerate(pred[i]):
                if pred[i, j]:
                    adj_matrix[ann, j+bbox_length] = 1
                    adj_matrix[j+bbox_length, ann] = 1

    return adj_matrix

def save_h5(final_features, adj_matrix, bbox_classes, disease_classes, test_topk_per_image, full=True, times=0, length = 100):
    filename = '/mnt/data/wy/PythonProject/EKAID/output/mimic_ana_box/feat_graph_labels.hdf5'
    if times == 0:
        h5f = h5py.File(filename, 'w')
        image_features_dataset = h5f.create_dataset("image_features", (length, test_topk_per_image, 1024),
                                                    maxshape=(None, test_topk_per_image, 1024),
                                                    chunks=(100, test_topk_per_image, 1024),
                                                    dtype='float32')
        image_adj_matrix_dataset = h5f.create_dataset("image_adj_matrix", (length, 100, 100),
                                                    maxshape=(None, 100, 100),
                                                    chunks=(100, 100, 100),
                                                    dtype='int64')
        bbox_classes_dataset = h5f.create_dataset("bbox_classes", (length, test_topk_per_image / 2),
                                                      maxshape=(None, test_topk_per_image / 2),
                                                      chunks=(100, test_topk_per_image / 2),
                                                      dtype='int64')
        disease_classes_dataset = h5f.create_dataset("disease_classes", (length, test_topk_per_image / 2, 38),
                                                  maxshape=(None, test_topk_per_image / 2, 138),
                                                  chunks=(100, test_topk_per_image / 2, 38),
                                                  dtype='int64')
    else:
        h5f = h5py.File(filename, 'a')
        image_features_dataset = h5f['image_features']
        image_adj_matrix_dataset = h5f['image_adj_matrix']
        bbox_classes_dataset = h5f['bbox_classes']
        disease_classes_dataset = h5f['disease_classes']


    if len(final_features) != length:
        adding = len(final_features)
    else:
        adding = length
    image_features_dataset.resize([times*length+adding, test_topk_per_image, 1024])
    image_features_dataset[times*length:times*length+adding] = final_features

    image_adj_matrix_dataset.resize([times*length+adding, 100, 100])
    image_adj_matrix_dataset[times*length:times*length+adding] = adj_matrix

    bbox_classes_dataset.resize([times*length+adding, test_topk_per_image / 2])
    bbox_classes_dataset[times*length:times*length+adding] = bbox_classes

    disease_classes_dataset.resize([times*length+adding, test_topk_per_image / 2, 38])
    disease_classes_dataset[times*length:times*length+adding] = disease_classes

    h5f.close()

if __name__ == '__main__':
    dataset = 'mimic'

    gold_weights=True
    if gold_weights:
        dd = get_kg2()  # len(dd) = 26
        category = {}
        for item in dd:
            category[item] = len(category)

    thing_classes = list(category)
    # category_name_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}
    category_name_to_id = category


    #===============================



    cfg = get_cfg()
    cfg.OUTPUT_DIR = 'results/mimic_ana_graph'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("mimic_ana_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-1-3000.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_pre.pth")  # this is the path of the trained vinbigdata model
    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(bboxlist)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    ### --- Inference & Evaluation ---
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = str("results/mimic_ana_bbx/model_final.pth")
    print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set a custom testing threshold
    print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    predictor = DefaultPredictor(cfg)

    # hook
    features = []
    proposals = []
    predictions = []

    # disease predictor
    model = Model(feat_dim=1024)
    model.load_state_dict(torch.load('results/mimic_disease/last_checkpoint.pth'))
    model.to('cuda')

    ### hook ##########
    layer_to_hook1 = 'roi_heads'
    # layer_to_hook2 = 'box_head'
    layer_to_hook2 = 'box_predictor'
    layer_to_hook3 = 'fc_relu2'
    for name, layer in predictor.model.named_modules():
        if name == layer_to_hook1:
            layer.register_forward_hook(save_features1)
            for name2, layer2 in layer.named_modules():
                if name2 == layer_to_hook2:
                    # for name3, layer3 in layer2.named_modules():
                    #     if name3 == layer_to_hook3:
                    layer2.register_forward_hook(save_features)


    if dataset == 'mimic':
        DatasetCatalog.register("mimic", get_mimic_dict)
        MetadataCatalog.get("mimic").set(thing_classes=bboxlist)
        dataset_dicts = get_mimic_dict(full=True)
    # for d in ["train", "val", "test"]:
    #     DatasetCatalog.register("vinbigdata_" + d, lambda d=d: get_dicts(d))
    #     MetadataCatalog.get("vinbigdata_" + d).set(thing_classes=['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity','Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis'])


    full = True
    #### test/val
    # metadata = MetadataCatalog.get("mimic")

    index = 0
    out_ids = []
    final_features = []
    bboxes = []
    normalized_bboxes = []
    pos_boxes = []
    # pred_classes = []
    bbox_classes = []
    disease_classes = []
    matrixes = []
    n = 0
    test_topk_per_image = 26

    batch_size = 1 # batch size must be dividable by length
    length = 1000
    times = 0
    flag = 0
    order = True
    if order:
        pre_extract_num = 100

    resume = False # remember to check before running
    if resume:
        stopped_batch_num = 377000  # the number you see in the terminal when stooped
        # stopped_batch_num = 75000
        stopped_img_num = stopped_batch_num * batch_size
        continue_i = (stopped_img_num - length) / batch_size
        times = int((stopped_img_num - length)/length)
        n = int(continue_i * test_topk_per_image)
    # miss_files = 0
    for i in tqdm(range(ceil(len(dataset_dicts) / batch_size))):
        # if i == 452:
        #     print('here')
        if resume:
            if i < continue_i:
                continue
        inds = list(range(batch_size * i, min(batch_size * (i + 1), len(dataset_dicts))))
        dataset_dicts_batch = [dataset_dicts[i] for i in inds]
        # print(dataset_dicts_batch[0]['file_name'])
        im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]
        # if im_list[0] is None:
        #     print(dataset_dicts_batch[0]['file_name'])
        #     continue
        #     miss_files += 1
        outputs_list = predict_batch(predictor, im_list)

        if order:
            out_ids = (
                inference(predictions[0], proposals[0], predictor.model.roi_heads.box_predictor.box2box_transform,
                          pre_extract_num))
        else:
            out_ids = (inference(predictions[0], proposals[0], predictor.model.roi_heads.box_predictor.box2box_transform, test_topk_per_image))
        predictions = []
        proposals = []
        for j, (feats, ids, dicts, outputs) in enumerate(zip(features, out_ids, dataset_dicts_batch,outputs_list)):
            ids = (ids / len(bbox_dict)).type(torch.long)
            if order:
                feat = []
                bb = []
                pred_class = []
                for id in range(test_topk_per_image):
                    for idx in range(pre_extract_num):
                        if outputs['instances']._fields['pred_classes'][idx] == id:
                            bb.append(outputs['instances']._fields['pred_boxes'].tensor[idx].cpu())
                            pred_class.append(outputs['instances']._fields['pred_classes'][idx].cpu())
                            feat.append(feats[ids[idx]].cpu())
                            break
                        elif idx == pre_extract_num-1:
                            bb.append(torch.zeros(4))
                            pred_class.append(torch.zeros(1))
                            feat.append(torch.zeros(1024))
                feat = torch.stack(feat)
                bb = torch.stack(bb)
                pred_class = torch.tensor(pred_class)

                img = im_list[j]
                disease_feats, disease_pred, matrix = predict_disease(model, img, bb, pred_class)

                final_features.append(np.array(torch.cat((feat, disease_feats.cpu()), dim=0)))
                # bb = np.array(bb)
                bbox_class = np.array(pred_class)
                disease_class = np.array(disease_pred.cpu())

            else:
                feat = feats[ids].cpu()
                final_features.append(np.array(feat))
                 # f[dicts['image_id']] = feat
                bb = np.array(outputs['instances']._fields['pred_boxes'].tensor.cpu())[:test_topk_per_image]
                pred_class = np.array(outputs['instances']._fields['pred_classes'].cpu())[:test_topk_per_image]
                img = im_list[j]
                disease_feats, disease_pred, matrix = predict_disease(model, img, bb, pred_class)
                bbox_class = pred_class
                disease_class = np.array(disease_pred)
            # bboxes.append(bb)
            bbox_classes.append(bbox_class)
            disease_classes.append(disease_class)
            # normalized_bb = np.concatenate((bb/1024,np.zeros((bb.shape[0],2))),1)
            # normalized_bboxes.append(normalized_bb)
            # pos_boxes.append([n, n+len(normalized_bb)])
            matrixes.append(matrix)
            n += len(matrixes)
        features = []
        if len(final_features) == length or i == ceil(len(dataset_dicts) / batch_size)-1:
            final_features = np.array(final_features)
            # bboxes = np.array(bboxes)
            # pred_classes = np.array(pred_classes)
            # normalized_bboxes = np.array(normalized_bboxes)
            # pos_boxes = np.array(pos_boxes)
            adj_matrix = np.array(matrixes)
            save_h5(final_features, adj_matrix, bbox_classes, disease_classes, test_topk_per_image*2, full=full, times= times, length=length)
            final_features = []
            bboxes = []
            normalized_bboxes = []
            pos_boxes = []
            # pred_classes = []
            bbox_classes = []
            disease_classes = []
            matrixes = []
            times += 1



    print('finished writing')

