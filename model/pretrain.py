import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json
import argparse
import time
import numpy as np
import torch

torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random
import h5py

from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.model_pretrain import ChangeDetector, AddSpatialInfo
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
    LanguageModelCriterion, decode_sequence, decode_beams, \
    build_optimizer, coco_gen_format_save, one_hot_encode, \
    EntropyLoss, load_checkpoint
from utils.mimic_utils import process_matrix
from tqdm import tqdm
from pycocotools.coco import COCO
from evaluation import my_COCOEvalCap
from transformers import BertTokenizer
import torch.distributed as dist

# os.environ['CUDA_LAUNCH_BLOCKING']='1'
# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_fold', type=str, default='mode2_location_all_0.0001_2022-09-16-22-43-34')
parser.add_argument('--snapshot', type=int, default=22000)
parser.add_argument('--checkpoint', type=str, default='')

parser.add_argument('--eval_target', type=str, default='test', choices=['test', 'val'])
parser.add_argument('--seed', type=int, default=1238)
parser.add_argument('--local_rank', type=int, default=-1)

args = parser.parse_args()
merge_cfg_from_file(args.cfg)

exp_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_' + str(args.seed)
print('Run name:', exp_name)

cfg.train.setting = args.setting

# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
# torch.backends.cudnn.enabled = False
# default_gpu_device = gpu_ids[0]
# torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")
# device = 'cpu'

# Experiment configuration
exp_dir = cfg.exp_dir
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, 'temp', exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = 'checkpoints/pretrained_models'
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = 'checkpoint_%d.pt'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('/mnt/data/wy/model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
tokenizer.add_special_tokens({'bos_token': '[DEC]'})
tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

# local_rank = args.local_rank
# dist.init_process_group(backend="nccl")
# local_rank = dist.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'test' if args.eval_target == 'test' else 'val')
annotation_file = 'data/replaced_dataset/mimic_gt_captions_test.json' if args.eval_target == 'test' else 'data/replaced_dataset/mimic_gt_captions_val.json'
train_size = len(train_dataset)
val_size = len(val_dataset)

resume = args.resume
if resume:
    print('loading checkpoints')
    snapshot_full_path = args.checkpoint
    assert os.path.exists(snapshot_full_path)
    checkpoint = load_checkpoint(snapshot_full_path)
    change_detector_state = checkpoint['model']
    # speaker_state = checkpoint['speaker_state']

    # Load modules
    change_detector = ChangeDetector(cfg, train_dataset.word_to_idx, tokenizer)
    change_detector.load_state_dict(change_detector_state)
    change_detector = change_detector.to(device)

    print('checkpoints loading successfully')
else:
    # Create model
    change_detector = ChangeDetector(cfg, train_dataset.word_to_idx, tokenizer)
    change_detector.to(device)
# DDP setting
# change_detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(change_detector)
# change_detector = torch.nn.parallel.DistributedDataParallel(change_detector, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


spatial_info = AddSpatialInfo()
spatial_info.to(device)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(spatial_info, file=f)

# freeze text embedding
for name, param in change_detector.text_encoder.named_parameters():
    if name=='embeddings':
        param.requires_grad = False
for name, param in change_detector.text_decoder.named_parameters():
    if name=='embeddings':
        param.requires_grad = False

# Define loss function and optimizer
optimizer = build_optimizer(filter(lambda p: p.requires_grad, change_detector.parameters()), cfg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)

# Train loop
t = 0
epoch = 0
best = 0

change_detector.train()

while t < cfg.train.max_iter:
    # while epoch < cfg.train.max_epoch:
    print('Starting epoch %d' % epoch)
    lr_scheduler.step()
    print(lr_scheduler.optimizer.param_groups[0]['lr'])
    speaker_loss_avg = AverageMeter()
    speaker_pos_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()

    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, sc_feats, labels, sc_pos_labels, masks, pair_index, d_adj_matrix, q_adj_matrix, question, seq_neg, keywords_sent = batch  # two stage
        batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
        d_adj_matrix, q_adj_matrix = d_adj_matrix.to(device), q_adj_matrix.to(device)
        d_adj_matrix = d_adj_matrix[:, :62, :62]
        q_adj_matrix = q_adj_matrix[:, :62, :62]
        optimizer.zero_grad()
        loss = change_detector(d_feats, sc_feats, d_adj_matrix, q_adj_matrix, question, labels, seq_neg, keywords_sent)

        # loss = loss.mean()
        stats = {}
        stats['loss'] = loss.item()
        loss.backward()
        optimizer.step()

        iter_end_time = time.time() - iter_start_time

        t += 1

        if t % cfg.train.log_interval == 0:
            print('epoch: {}/{}'.format(epoch, cfg.train.max_epoch), '[{}/{}]'.format(i * batch_size, train_size),
                  'loss: ', stats, int(iter_end_time * 60), 's')

        # Evaluation
        if t % cfg.train.snapshot_interval == 0:
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'model': chg_det_state,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % t)
            save_checkpoint(checkpoint, save_path)
    epoch += 1
