import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer
import math
from models.med import BertConfig, BertModel, BertLMHeadModel, BertForMaskedLM
import time
import torchvision.models as tmodels
from models.mccformer import MCCFormers_D, MCCFormers_S
import utils.gloria_loss as gloria
import random
from models.graph_embedding import GraphEmbedding
class ChangeDetector(nn.Module):
    def __init__(self, cfg, word_to_idx, tokenizer, vocab_size=148,
                 med_config='configs/bert_config.json'):
        super().__init__()
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.dim
        self.feat_dim = cfg.model.change_detector.feat_dim - 2
        self.att_head = cfg.model.change_detector.att_head
        self.att_dim = cfg.model.change_detector.att_dim
        self.nongt_dim = cfg.model.change_detector.nongt_dim
        self.pos_emb_dim = cfg.model.change_detector.pos_emb_dim

        self.img = nn.Linear(self.feat_dim, 768)

        # =====================ViT====================
        # self.visual_encoder = AutoModel.from_pretrained('/mnt/data/wy/model/BiomedCLIP-PubMedBert_256-vit_base_path16_224')
        # self.visual_encoder, vision_width = create_vit(vit='base',image_size=1024, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0)

        # =====================Bert===================
        # Bert Encoder&tokenizer
        self.tokenizer = tokenizer
        encoder_config = BertConfig.from_json_file(med_config)
        # encoder_config.encoder_width = self.att_dim
        self.text_encoder = BertModel.from_pretrained(
            '/mnt/data/wy/model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            config=encoder_config, add_pooling_layer=False, ignore_mismatched_sizes=True)
        # self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.att_dim)

        decoder_config = BertConfig.from_json_file(med_config)
        # decoder_config.encoder_width = self.att_dim

        # LM Bert Text Decoder
        # self.text_decoder = BertLMHeadModel.from_pretrained('/mnt/data/wy/model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        #                                               config=decoder_config, ignore_mismatched_sizes=True)
        # self.text_decoder = BertLMHeadModel(config=decoder_config)

        # MLM Bert Text Decoder
        self.text_decoder = BertForMaskedLM.from_pretrained(
            '/mnt/data/wy/model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            config=decoder_config, ignore_mismatched_sizes=True)
        # self.text_decoder = BertForMaskedLM(config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # Diff Bert Decoder
        # self.text_diff_decoder = BertForMaskedLM(config=decoder_config)
        # self.text_diff_decoder.resize_token_embeddings(len(self.tokenizer))

        # parameter share
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

        self.contrastive_proj = nn.Linear(768, self.att_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # self.mccformer = MCCFormers_D(768, dropout=0.5, d_model=768, n_head=4, n_layers=2)
        self.mccformer = MCCFormers_S(768, d_model=768, n_head=4, n_layers=2)

        self.cfg = cfg
        text_width = self.text_encoder.config.hidden_size
        self.itm_head = nn.Linear(text_width, 2)
        self.question_head = nn.Linear(text_width, 2)

        self.graph_embedding = GraphEmbedding(self.text_encoder, self.tokenizer)
        # self.classifier = nn.Linear(768, 487)

    def mask(self, input_ids, vocab_size, device, targets=None, probability_matrix=None):
        # 关键词ids
        mask_area = input_ids > 39
        masked_indices = []
        for i, mask_row in enumerate(mask_area):
            len = (mask_row == True).sum()
            if len == 1:
                # 关键词数量为1时，全部mask
                masked_indices.append(mask_area[i])
            elif len > 1:
                # 关键词数量大于1时，mask60%关键词
                probability_matrix = torch.zeros_like(mask_area[i], dtype=torch.float32).masked_fill(mask_area[i], 0.6)
                masked_indices.append(torch.bernoulli(probability_matrix).bool())
            else:
                # 不存在关键词，mask30%全句单词
                probability_matrix = torch.full(mask_area[i].shape, 0.3)
                masked_indices.append(torch.bernoulli(probability_matrix).bool())

        masked_indices = torch.stack(masked_indices)

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8).cuda()).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5).cuda()).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward(self, input_1, input_2, d_adj_matrix, q_adj_matrix, question, answer, seq_neg, keywords_sent):

        batch_size = input_1.size(0)

        # input_bef = self.visual_encoder(input_1).last_hidden_state[:,1:,:]
        # input_aft = self.visual_encoder(input_2).last_hidden_state[:,1:,:]

        input_bef = self.img(input_1)
        input_aft = self.img(input_2)

        graph_bef = self.graph_embedding(d_adj_matrix.float())
        graph_aft = self.graph_embedding(q_adj_matrix.float())

        input_bef = torch.cat((input_bef, graph_bef), dim=1)
        input_aft = torch.cat((input_aft, graph_aft), dim=1)

        # ===============for Bert(text)=============
        question_token = self.tokenizer(question, padding='max_length', truncation=True, max_length=30,
                                        return_tensors='pt').to(input_1.device)
        answer_token = self.tokenizer(answer, padding='max_length', truncation=True, max_length=90,
                                      return_tensors='pt').to(input_1.device)
        neg_answer_token = self.tokenizer(seq_neg, padding='max_length', truncation=True, max_length=90,
                                          return_tensors='pt').to(input_1.device)
        keywords_token = self.tokenizer(keywords_sent, padding='max_length', truncation=True, max_length=90, return_tensors='pt').to(input_1.device)


        encoder_input_ids = question_token.input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # ==============================Cross Attention===============================
        hidden_input_bef = self.mccformer(input_bef, input_aft - input_bef)
        hidden_input_aft = self.mccformer(input_aft, input_aft - input_bef)
        image_input = torch.cat((hidden_input_bef, hidden_input_aft), dim=1)
        question_embed = self.text_encoder(encoder_input_ids, attention_mask=question_token.attention_mask,
                                           encoder_hidden_states=image_input, return_dict=True)

        # text_embed = self.text_encoder(encoder_input_ids.clone(), attention_mask=question_token.attention_mask,
        #                                return_dict=True, mode='text')
        # question_choice = self.question_head(text_embed.last_hidden_state[:, 0, :])
        # question_cls = torch.argmax(question_choice, dim=-1).bool()

        # =================================ITC====================================
        with torch.no_grad():
            text_sim = torch.zeros(batch_size, batch_size).to(input_1.device)
            keywords_token_ids = keywords_token.input_ids.clone()
            for i in range(keywords_token_ids.size(0)):
                for j in range(i, keywords_token_ids.size(0)):
                    if keywords_token_ids[i].equal(keywords_token_ids[j]):
                        text_sim[i, j] = 1
            self.temp.clamp_(0.001,0.5)
            gt_matrix = text_sim

        image_feat = self.contrastive_proj(question_embed.last_hidden_state[:, 0, :])
        keywords_len = keywords_token.attention_mask.sum(dim=1) - 2
        keywords_embeds = self.text_encoder.embeddings(keywords_token.input_ids)[:, 1:, :]
        local_loss = gloria.local_loss(question_embed.last_hidden_state[:, 1:, :], keywords_embeds, keywords_len, gt_matrix)
        global_loss = gloria.global_loss(image_feat, torch.mean(keywords_embeds, dim=1), gt_matrix)
        loss_cl = local_loss + global_loss

        # ===============================ITM===================================
        # positive forward
        itm_input_ids = answer_token.input_ids.clone()
        itm_input_ids[:,0] = self.tokenizer.enc_token_id
        pos_output = self.text_encoder(itm_input_ids, attention_mask=answer_token.attention_mask,
                                       encoder_hidden_states=question_embed.last_hidden_state, return_dict=True)
        # negative forward
        neg_input_ids = neg_answer_token.input_ids.clone()
        neg_output = self.text_encoder(neg_input_ids, attention_mask=neg_answer_token.attention_mask,
                                       encoder_hidden_states=question_embed.last_hidden_state, return_dict=True)
        vl_embeddings = torch.cat([pos_output.last_hidden_state[:,0,:], neg_output.last_hidden_state[:,0,:]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)],
                               dim=0).to(input_1.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        # ===============================MLM===================================

        # TODO 关键词MLM
        decoder_input_ids = answer_token.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        decoder_input_ids, labels = self.mask(decoder_input_ids, self.text_decoder.config.vocab_size, input_1.device,
                                              targets=labels, probability_matrix=probability_matrix)

        mlm_output = self.text_decoder(decoder_input_ids, attention_mask=answer_token.attention_mask,
                                       encoder_hidden_states=question_embed.last_hidden_state, labels=labels,
                                       return_dict=True, reduction='none')

        # mlm_diff_output = self.text_diff_decoder(decoder_input_ids, attention_mask=answer_token.attention_mask,
        #                                          encoder_hidden_states=question_embed.last_hidden_state, labels=labels,
        #                                          return_dict=True, reduction='none')
        # mlm_output = mlm_output.loss * question_cls.float()
        # mlm_diff_output = mlm_diff_output.loss * (~question_cls).float()
        # loss_mlm = torch.cat((mlm_output, mlm_diff_output), dim=0)
        # loss_mlm = torch.mean(loss_mlm) * 2
        loss_mlm = mlm_output.loss

        return loss_mlm + loss_cl + loss_itm
class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug


from typing import List


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            skip_key: str,
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            # print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)