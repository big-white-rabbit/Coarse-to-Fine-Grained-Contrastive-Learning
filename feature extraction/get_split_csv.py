import os
import json
import pandas as pd
from tqdm import tqdm
import csv

def save(split, ana_dir='/mnt/data/wy/dataset/silver_dataset/annotation',
             split_dir='/mnt/data/wy/dataset/silver_dataset/splits',
             img_dir='/mnt/data/wy/dataset/mimic_cxr_png'):
    ana_files = []
    ana_dict = {}
    csv_data = []
    csv_file = open(split+'_ann.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['img_path', 'x1', 'y1', 'x2', 'y2', 'bbox_label', 'disease_label'])
    with open(os.path.join(split_dir, split+'.csv'), 'r') as f:
        df = pd.read_csv(f)
        json_files = [image_id + '.json' for image_id in df['dicom_id']]
    for json_file in tqdm(json_files):
        file = os.path.join(ana_dir, json_file)
        if os.path.exists(file):
            with open(file, 'r') as f:
                data = json.load(f)
                img_path = os.path.join(img_dir, data['image_id']+'.png')
                for object in data['annotations']:
                    bbox = object['bbox_224']
                    bbox = [x / float(224/1024) for x in bbox]
                    bbox_label = object['bbox_name']
                    disease_label = ''
                    if len(object['disease_name']) > 0:
                        for i, disease in enumerate(object['disease_name']):
                            if i != len(object['disease_name'])-1:
                                disease_label += (disease+'|')
                            else:
                                disease_label += disease
                    csv_data.append(img_path)
                    csv_data.append(bbox[0])
                    csv_data.append(bbox[1])
                    csv_data.append(bbox[2])
                    csv_data.append(bbox[3])
                    csv_data.append(bbox_label)
                    csv_data.append(disease_label)
                    writer.writerow(csv_data)
                    csv_data = []

if __name__ == '__main__':
    splits = ['train', 'valid']
    for split in splits:
        save(split)