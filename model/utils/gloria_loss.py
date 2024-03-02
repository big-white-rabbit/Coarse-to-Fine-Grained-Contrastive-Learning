"""
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn

from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x queryL x ndf

    context: batch x sourceL x ndf (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(1)
    query = torch.transpose(query, 1, 2).contiguous()
    # ih, iw = context.size(2), context.size(3)
    sourceL = context.size(1)

    # --> batch x sourceL x ndf
    contextT = context  # batch_size, sourceL, n_dim
    context = torch.transpose(context, 1, 2).contiguous() # batch_size, n_dim, sourceL

    # context = context.view(batch_size, -1, sourceL)   # batch_size, n_dim, sourceL
    # contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, sourceL)


def global_loss(cnn_code, rnn_code, labels, eps=1e-8, temp3=10.0):
    # cnn_code  batch_size, cnn_len, dim
    # rnn_code  batch_size, rnn_len, dim
    batch_size = cnn_code.shape[0]

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.BCEWithLogitsLoss()(scores0, labels)
    loss1 = nn.BCEWithLogitsLoss()(scores1, labels.transpose(0, 1))
    return (loss0 + loss1) / 2


def local_loss(
    img_features, words_emb, cap_lens, labels, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):

    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :words_num, :].unsqueeze(0).contiguous()  # [1, len, 768]
        word = word.repeat(batch_size, 1, 1)  # [48, len, 768]
        context = img_features  # [batch, len, 768]

        weiContext, attn = attention_fn(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 90]

        # att_maps.append(
        #     attn[i].unsqueeze(0).contiguous()
        # )  # add attention for curr index  [25, 19, 19]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    loss0 = nn.BCEWithLogitsLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.BCEWithLogitsLoss()(similarities1, labels.transpose(0, 1))
    return (loss0 + loss1) / 2