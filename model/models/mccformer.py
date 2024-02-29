import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_

class CrossTransformer(nn.Module):
  """
  Cross Transformer layer
  """
  def __init__(self, dropout, d_model = 512, n_head = 4):
    """
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    """
    super(CrossTransformer, self).__init__()
    self.attention = nn.MultiheadAttention(d_model, n_head, dropout = dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.activation = nn.ReLU()

    self.linear1 = nn.Linear(d_model, d_model * 4)
    self.linear2 = nn.Linear(d_model * 4, d_model)

  def forward(self, input1, input2):
    attn_output, attn_weight = self.attention(input1, input2, input2)
    output = input1 + self.dropout1(attn_output)
    output = self.norm1(output)
    ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
    output = output + self.dropout3(ff_output)
    output = self.norm2(output)

    return output

class MCCFormers_D(nn.Module):
    def __init__(self, feature_dim, dropout, d_model=512, n_head=4, n_layers=2):
        super(MCCFormers_D, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        self.projection = nn.Linear(feature_dim, d_model)
        self.position_embedding = nn.Embedding(114, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input1, input2):
        # input.size: (batch_size, n_features(52), feature_dim)
        # print(input1.shape)
        batch = input1.size(0)
        n = input1.size(1)
        dim = input1.size(-1)

        # cls1 = torch.zeros((batch, 1, dim)).to(input1.device)
        # cls2 = torch.zeros((batch, 1, dim)).to(input1.device)
        input1 = self.projection(input1).permute(1, 0, 2)   # (n_features, batch_size, d_model)
        input2 = self.projection(input2).permute(1, 0, 2)
        position = torch.arange(n).to(input1.device)   # n_features
        position_embed = self.position_embedding(position)  # n_features, d_model
        position_embed = position_embed.unsqueeze(1).repeat(1, batch, 1)  # n_features, batch_size, d_model

        output1 = input1 + position_embed
        output2 = input2 + position_embed
        for layer in self.transformer:
            output1, output2 = layer(output1, output2), layer(output2, output1)

        # output = (output2 - output1).permute(1, 0, 2) # batch_size, n_features, d_model
        # return output1.permute(1, 0, 2), output2.permute(1, 0, 2)
        return torch.cat((output1, output2), dim=0).permute(1, 0, 2)

class MCCFormers_S(nn.Module):
    def __init__(self, feature_dim, d_model=512, n_head=4, n_layers=2, dim_feedforward=2048):
        super(MCCFormers_S, self).__init__()

        self.d_model = d_model
        self.projection = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.idx_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(88, d_model)

    def forward(self, input1, input2):
        # input batch_size, n_features(52), feature_dim
        batch = input1.size(0)
        n = input1.size(1)

        input1 = self.projection(input1)    # batch_size, n_features, d_model
        input2 = self.projection(input2)
        position = torch.arange(n).to(input1.device)    # n_features(52)
        position_embed = self.position_embedding(position).repeat(batch, 1, 1)  # batch_size, n_features, d_model
        input = torch.cat([input1+position_embed, input2+position_embed], dim=1)    # batch_size, 2*n_features, d_model
        input = input.permute(1, 0, 2)  # 2*n_features, batch_size, d_model

        idx1 = torch.zeros(batch, n).long().to(input1.device)  # batch_size, n_features
        idx2 = torch.ones(batch, n).long().to(input2.device)    # batch_size, n_features
        idx = torch.cat([idx1, idx2], dim=1)            # batch_size, 2*n_features
        idx_embed = self.idx_embedding(idx)                     # batch_size, 2*features, d_model
        idx_embed = idx_embed.permute(1, 0, 2)  # 2*n_features, batch_size, d_model

        output = self.transformer(input+idx_embed)              # 2*features, batch_size, d_model

        output1 = output[:n,:,:].permute(1, 0, 2)   # batch_size, n_features, d_model
        output2 = output[n:,:,:].permute(1, 0, 2)   # batch_size, n_features, d_model
        # return output1, output2
        # return output2-output1
        return output.permute(1, 0, 2)