import torch
import os
import pandas as pd
import json
import numpy as np
import random
random.seed(888888)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from torch import nn
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

weights = torch.tensor([5786499/764482.0, 5786499/280655.0, 5786499/262679.0,
                        5786499/29565.0, 5786499/60391.0, 5786499/27134.0,
                        5786499/58200.0, 5786499/28859.0, 5786499/7529.0,
                        5786499/37711.0, 5786499/20910.0, 5786499/172363.0,
                        5786499/15921.0, 5786499/112172.0, 5786499/28256.0,
                        5786499/38918.0, 5786499/7335.0, 5786499/43772.0,
                        5786499/3299.0, 5786499/25738.0, 5786499/15748.0,
                        5786499/10665.0, 5786499/27121.0, 5786499/11462.0,
                        5786499/16119.0, 5786499/5411.0, 5786499/7048.0,
                        5786499/6316.0, 5786499/3297.0, 5786499/2219.0,
                        5786499/9077.0, 5786499/3227.0, 5786499/605.0,
                        5786499/4047.0, 5786499/9865.0, 5786499/2664.0,
                        5786499/3192, 5786499/2391], dtype=torch.float32)

class MIMIC_Dataset(Dataset):
    def __init__(self, split, ana_dir='/mnt/data/wy/dataset/silver_dataset/annotation',
                 split_dir='/mnt/data/wy/dataset/silver_dataset/splits',
                 img_dir='/mnt/data/wy/dataset/mimic_cxr_png'):
        super().__init__()
        self.ana_dir = ana_dir
        self.split_dir = split_dir
        self.img_dir = img_dir
        self.ana_files = []
        normal_object = 0
        with open(os.path.join(split_dir, split+'.csv'), 'r') as f:
            df = pd.read_csv(f)
            json_files = [image_id + '.json' for image_id in df['dicom_id']]

        print('processing datasets......')
        for json_file in tqdm(json_files):
            file = os.path.join(self.ana_dir, json_file)
            if os.path.exists(file):
                with open(file, 'r') as f:
                    data = json.load(f)
                    img_path = os.path.join(self.img_dir, data['image_id']+'.png')
                    for object in data['annotations']:
                        ana_dict = {}
                        if object['bbox_name'] not in bbox_dict:
                            continue

                        bbox = object['bbox_224']
                        bbox = [x / float(224/1024) for x in bbox]

                        bbox_label = object['bbox_name']
                        disease_label = object['disease_name']

                        if len(disease_label)==0 and random.random()>0.2:
                            continue
                        if len(disease_label) == 0:
                            normal_object += 1
                        ana_dict['img_path'] = img_path
                        ana_dict['bbox'] = bbox
                        ana_dict['bbox_label'] = bbox_label
                        ana_dict['disease_label'] = disease_label
                        self.ana_files.append(ana_dict)
        print('size of {} dataset: {}'.format(split, len(self.ana_files)))
        print('num of normal object of {} dataset: {}'.format(split, normal_object))

    def __len__(self):
        return len(self.ana_files)

    def __getitem__(self, index):
        data = self.ana_files[index]
        img_path = data['img_path']
        bbox = data['bbox']
        bbox_label = data['bbox_label']
        disease_label = data['disease_label']
        img = Image.open(img_path).convert('RGB')

        resize = transforms.Resize((224, 224))
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img = toTensor(img)
        if (int(bbox[2])-int(bbox[0])) * (int(bbox[3])-int(bbox[1])) == 0:
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            img = img[:, int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
            img = resize(img)
            img = normalize(img)
        label = torch.zeros(len(diseaselist), dtype=torch.float32)
        if len(disease_label) > 0:
            binary_label = 1
            for disease in disease_label:
                if disease not in disease_dict:
                    continue
                label[disease_dict[disease]] = 1
        else:
            binary_label = 0

        return img, bbox_dict[bbox_label], label, binary_label

class Model(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.embedding = nn.Embedding(len(bboxlist), feat_dim)
        self.net = models.resnet101()
        fc_inp = self.net.fc.in_features
        self.net.fc = nn.Linear(fc_inp, feat_dim)
        self.binary_classifier = nn.Linear(feat_dim, 2)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, len(diseaselist)),
            nn.Sigmoid()
        )

    def forward(self, img, bbox_label):
        pos_embed = self.embedding(bbox_label)
        img_feat = self.net(img)
        output = self.classifier(img_feat+pos_embed)
        binary_output = self.binary_classifier(img_feat+pos_embed)

        return output, binary_output

    def _forward(self, img, bbox_label):
        pos_embed = self.embedding(bbox_label)
        img_feat = self.net(img)
        output = self.classifier(img_feat+pos_embed)
        binary_output = self.binary_classifier(img_feat+pos_embed)

        return img_feat, output, binary_output


def precision(label, pred, binary_label, binary_pred, threshold=0.4):
    _, binary_pred = torch.max(binary_pred, 1)
    binary_label = binary_label.cpu().detach().numpy()
    binary_pred = binary_pred.cpu().detach().numpy()

    res = binary_label==binary_pred
    binary_acc = sum(res) / len(res)

    # n = 0
    # precision_acc_sum, recall_acc_sum, f1_acc_sum = 0.0, 0.0, 0.0
    # for i in range(len(binary_label)):
    #     if binary_label[i] == 1:
    #         pred = pred.ge(threshold).int()
    #         label = label.int()
    #         label = label.cpu().detach().numpy()
    #         pred = pred.cpu().detach().numpy()
    #         precision_acc = precision_score(label[i], pred[i])
    #         recall_acc = recall_score(label[i], pred[i])
    #         f1_acc = f1_score(label[i], pred[i])
    #         precision_acc_sum += precision_acc
    #         recall_acc_sum += recall_acc
    #         f1_acc_sum += f1_acc
    #         n += 1
    # if n != 0:
    #     recall_acc_sum = precision_acc_sum/n
    #     recall_acc_sum = recall_acc_sum/n
    #     f1_acc_sum = f1_acc_sum/n

    pred = pred.ge(threshold).int()
    label = label.int()
    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    acc = 0.0
    for i in range(len(binary_label)):
        res1 = label[i] == pred[i]
        if sum(res1)==len(label[i]):
            acc += 1.0
        else:
            p = sum(np.logical_and(label[i], pred[i]))
            q = sum(np.logical_or(label[i], pred[i]))
            acc += p / q
    acc = acc / len(binary_label)



    # return precision_acc_sum, recall_acc_sum, f1_acc_sum, binary_acc
    return acc, binary_acc


def train(device, batch_size, epochs=10, interval=50, resume=False):
    train_dataset = MIMIC_Dataset('train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    valid_dataset = MIMIC_Dataset('valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)

    model = Model(feat_dim=1024)
    if resume:
        model.load_state_dict(torch.load('results/mimic_disease/last_checkpoint.pth'))
        print('last checkpoint resumed')
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000001, weight_decay=0.00001)
    criterion = torch.nn.BCELoss()
    binary_criterion = torch.nn.CrossEntropyLoss()

    best_acc, n = 0.0, 0

    for epoch in range(epochs):
        model.train()
        train_l_sum, acc_sum, binary_acc_sum = 0.0, 0.0, 0.0
        for i, (img, bbox_label, label, binary_label) in enumerate(train_dataloader):
            img, bbox_label, label, binary_label = img.to(device), bbox_label.to(device), label.to(device), binary_label.to(device)
            # label = label.to(torch.float32)
            pred, binary_pred = model(img, bbox_label)
            loss = criterion(pred, label)
            loss_binary = binary_criterion(binary_pred, binary_label)

            loss = loss+loss_binary

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1

            train_l_sum += loss.item()
            acc, binary_acc = precision(label, pred, binary_label, binary_pred)
            binary_acc_sum += binary_acc
            acc_sum += acc

            if n%interval == 0:
                print('epoch: {}/{}'.format(epoch, epochs), '[{}/{}]'.format(i, len(train_dataloader)), 'loss: ', train_l_sum / interval,
                      'binary_acc: ', binary_acc_sum / interval,'acc: ', acc_sum / interval)
                train_l_sum, binary_acc_sum, acc_sum = 0.0, 0.0, 0.0

            if n%1000 == 0:
                print('save checkpoint at iter: ', n)
                torch.save(model.state_dict(), 'results/mimic_disease/last_checkpoint.pth')
                print('evaluating at iter: ', n)
                model.eval()
                with torch.no_grad():
                    acc_sum, binary_acc_sum = 0.0, 0.0
                    for i, (img, bbox_label, label, binary_label) in tqdm(enumerate(valid_dataloader)):
                        img, bbox_label, label, binary_label = img.to(device), bbox_label.to(device), label.to(device), binary_label.to(device)
                        # label = label.to(torch.float32)
                        pred, binary_pred = model(img, bbox_label)

                        acc, binary_acc = precision(label, pred, binary_label, binary_pred)

                        acc_sum += acc
                        binary_acc_sum += binary_acc

                    acc_avg = acc_sum/len(valid_dataloader)
                    binary_acc_avg = binary_acc_sum/len(valid_dataloader)

                    eval_score = {}
                    eval_score['iter'] = n
                    eval_score['acc'] = acc_avg
                    eval_score['binary_acc'] = binary_acc_avg

                    with open('results/mimic_disease/eval_score.json', 'a', newline='\n') as f:
                        data = json.dumps(eval_score, indent=4)
                        f.write(data)

                    print('acc: ', acc_avg,
                          'binary_acc: ', binary_acc_avg)
                    if binary_acc_avg*acc_avg > best_acc:
                        print('best acc: {}, save best model!'.format(binary_acc_avg*acc_avg))
                        torch.save(model.state_dict(), 'results/mimic_disease/best_checkpoint_bef.pth')

if __name__ == '__main__':
    train(device='cuda', batch_size=128, interval=50, resume=True)
