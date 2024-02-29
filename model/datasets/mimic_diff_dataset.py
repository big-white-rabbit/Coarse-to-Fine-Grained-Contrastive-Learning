import os
import json
import numpy as np
import random
import time

import h5py
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image

class RCCDataset_mimic(Dataset):

    def __init__(self, cfg, split):
        self.cfg = cfg

        # print('Speaker Dataset loading vocab json file: ', cfg.data.vocab_json)
        self.vocab_json = cfg.data.vocab_json
        self.word_to_idx = json.load(open(self.vocab_json, 'r'))
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.idx_to_word)+1
        # print('vocab size is ', self.vocab_size)

        self.splits = json.load(open(cfg.data.splits_json, 'r'))
        self.split = split
        self.keywords = json.load(open('./data/keywords.json', 'r'))

        if split == 'train':
            self.batch_size = cfg.data.train.batch_size
            self.seq_per_img = cfg.data.train.seq_per_img
            self.split_idxs = self.splits['train']
            self.num_samples = len(self.split_idxs)
            if cfg.data.train.max_samples is not None:
                self.num_samples = min(cfg.data.train.max_samples, self.num_samples)
        elif split == 'val':
            self.batch_size = cfg.data.val.batch_size
            self.seq_per_img = cfg.data.val.seq_per_img
            self.split_idxs = self.splits['val']
            self.num_samples = len(self.split_idxs)
            if cfg.data.val.max_samples is not None:
                self.num_samples = min(cfg.data.val.max_samples, self.num_samples)
        elif split == 'test':
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = self.splits['test']
            self.num_samples = len(self.split_idxs)
            if cfg.data.test.max_samples is not None:
                self.num_samples = min(cfg.data.test.max_samples, self.num_samples)
        elif split == 'all':
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = self.splits['train'] + self.splits['val'] + self.splits['test']
            self.num_samples = len(self.split_idxs)
        else:
            raise Exception('Unknown data split %s' % split)

        # print("Dataset size for %s: %d" % (split, self.num_samples))

        # load in the sequence data
        self.h5_label_file = h5py.File(cfg.data.h5_label_file, 'r')
        self.labels = self.h5_label_file['answers'][:] # just gonna load...
        self.questions = self.h5_label_file['questions'][:]
        # self.neg_labels = self.h5_label_file['neg_labels'][:]
        self.pos = self.h5_label_file['pos'][:]
        # self.pos = self.h5_label_file['answers'][:][:,:20]
        seq_size = self.labels.shape
        # self.neg_pos = self.h5_label_file['neg_pos'][:]
        self.max_seq_length = seq_size[1]
        # self.label_start_idx = self.h5_label_file['label_start_idx'][:]
        # self.label_end_idx = self.h5_label_file['label_end_idx'][:]
        self.label_start_idx = np.arange(len(self.pos)).reshape(len(self.pos),1)
        self.label_end_idx = self.label_start_idx+1
        self.feature_idx = self.h5_label_file['feature_idx'][:]
        # self.neg_label_start_idx = self.h5_label_file['neg_label_start_idx'][:]
        # self.neg_label_end_idx = self.h5_label_file['neg_label_end_idx'][:]
        # print('Max sequence length is %d' % self.max_seq_length)
        self.h5_label_file.close()

        # path4 = os.path.join('/mnt/data/wy/PythonProject/EKAID/output/mimic_ana_box/feat_graph.hdf5')
        path4 = os.path.join('data', '../data/cmb_bbox_di_feats.hdf5')
        self.hf4 = h5py.File(path4, 'r')
        self.features4 = self.hf4['image_features']
        self.node_one_num = int(len(self.features4[0])/2)
        assert (self.node_one_num == 26) # or go to modify nongt_dim
        # self.adj = self.hf4['image_adj_matrix']
        self.adj = self.hf4['semantic_adj_matrix']

    def get_image(self, index):
        study_id = self.pd.iloc[index]['study_id']
        dicom_id = self.mimic_all[self.mimic_all['study_id'] == int(study_id)]['dicom_id'].values[0]
        file_path = '/mnt/data/wy/dataset/mimic_cxr_png/%s.png'%str(dicom_id)
        image = Image.open(file_path).convert('RGB')
        transform = transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                             (0.229, 0.224, 0.225))
                                        ])
        image = transform(image)
        # image = image.resize((224,224))
        # image = np.array(image)
        return image

    def decode_sequence(self, seq):
        ix_to_word = self.idx_to_word
        txt = ''
        for j in range(seq.size):
            ix = seq[j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix.item()]
            else:
                break
        return txt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        random.seed(1111)
        img_idx = self.split_idxs[index] # pair index

        # feature order: ana, location
        d_feature = self.features4[self.feature_idx[img_idx, 0]]
        q_feature = self.features4[self.feature_idx[img_idx, 1]]
        d_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 0]]).double()
        q_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 1]]).double()

        # Fetch sequence labels
        ix1 = self.label_start_idx[img_idx]
        ix2 = self.label_end_idx[img_idx]
        n_cap = ix2 - ix1 + 1

        seq = np.zeros([self.seq_per_img, self.max_seq_length + 1], dtype=int)
        pos = np.zeros([self.seq_per_img, self.max_seq_length + 1], dtype=int)
        if n_cap < self.seq_per_img:
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :self.max_seq_length] = \
                    self.labels[ixl, :self.max_seq_length]
                pos[q, :self.max_seq_length] = \
                    self.pos[ixl, :self.max_seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            seq[:, :self.max_seq_length] = \
                self.labels[ixl: ixl + self.seq_per_img, :self.max_seq_length]
            pos[:, :self.max_seq_length] = \
                self.pos[ixl: ixl + self.seq_per_img, :self.max_seq_length]

        # Generate masks
        mask = np.zeros_like(seq)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, seq)))
        for ix, row in enumerate(mask):
            row[:nonzeros[ix]] = 1

        # neg_mask = np.zeros_like(neg_seq)
        # nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, neg_seq)))
        # for ix, row in enumerate(neg_mask):
        #     row[:nonzeros[ix]] = 1
        question = self.questions[ixl]
        seq = self.decode_sequence(np.squeeze(seq,axis=0)[1:])
        seq_neg = self.get_neg_seq(seq)
        question = self.decode_sequence(question)
        keywords_sent = self.get_keywords_sent(seq, question)



        return (d_feature, q_feature, seq, pos,  mask, img_idx, d_adj_matrix, q_adj_matrix, question, seq_neg, keywords_sent)

    def get_neg_seq(self, seq):
        #keywords
        abnormality_words = self.keywords[0]['abnormality_words']
        presence_words = self.keywords[1]['presence_words']
        view_words = self.keywords[2]['view_words']
        location_words = self.keywords[3]['location_words']
        level_words = self.keywords[4]['level_words']

        for word in abnormality_words:
            if word in seq:
                seq = seq.replace(word, random.choice(abnormality_words))
            return seq
        for word in presence_words:
            if word in seq:
                return 'no' if (word=='yes' and random.uniform(0,1) > 0.2) else 'no'
        for word in view_words:
            if word in seq:
                return 'PA view' if (word=='AP view' and random.uniform(0,1) > 0.2) else 'AP view'
        for word in location_words:
            if word in seq:
                seq = seq.replace(word, random.choice(location_words))
            return seq
        for word in level_words:
            if word in seq:
                seq = seq.replace(word, random.choice(level_words))
            return seq
        return seq

    def get_keywords_sent(self, seq, question):
        if question.startswith('what has changed'):
            seq = seq.replace('the main image has ', '')
            seq = seq.replace('the main image is ', '')
            seq = seq.replace('than the reference image', '')
            seq = seq.replace('finding of ', '')
            seq = seq.replace('findings of ', '')
            seq = seq.replace('the level of ', '')
            seq = seq.replace('an ', '')
            seq = seq.replace('the ', '')
            return seq
        else:
            return seq

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length


def rcc_collate(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]
    q_feat_batch = transposed[1]
    seq_batch = default_collate(transposed[2])
    pos_batch = default_collate(transposed[3])
    mask_batch = default_collate(transposed[4])
    index_batch = default_collate(transposed[5])
    d_adj_matrix = default_collate(transposed[6])
    q_adj_matrix = default_collate(transposed[7])
    question = default_collate(transposed[8])
    seq_neg = default_collate(transposed[9])
    keywords_sent = default_collate(transposed[10])
    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)
    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    # d_img_batch = transposed[11]
    # n_img_batch = transposed[12]
    # q_img_batch = transposed[13]
    return (d_feat_batch, q_feat_batch,
            seq_batch, pos_batch,
            mask_batch,index_batch,d_adj_matrix,q_adj_matrix,question, seq_neg, keywords_sent)


class RCCDataLoader(DataLoader):

    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = rcc_collate
        super().__init__(dataset, **kwargs)


if __name__=='__main__':
    from configs.config import cfg, merge_cfg_from_file
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--entropy_weight', type=float, default=0.0)
    parser.add_argument('--visualize_every', type=int, default=10)
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_fold', type=str, default='mode2_location_0.0001_2022-05-19-00-18-37')
    parser.add_argument('--snapshot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1113)

    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)

    dataset = RCCDataset_mimic(cfg, 'test')
    print(dataset[10])