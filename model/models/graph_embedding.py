import torch
from torch import nn
import math

bboxlist = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone',
            'right hilar structures', 'right apical zone', 'right costophrenic angle',
            'right hemidiaphragm', 'left lung', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone',
            'left hilar structures', 'left apical zone', 'left costophrenic angle', 'left hemidiaphragm',
            'spine', 'aortic arch', 'mediastinum', 'upper mediastinum',
            'cardiac silhouette', 'left cardiac silhouette', 'right cardiac silhouette', 'cavoatrial junction'
]

diseaselist = ['lung opacity','pleural effusion','atelectasis','lobar/segmental collapse最常见肺不张(atelectasis)肺炎(pneumonia)症状','enlarged cardiac silhouette','airspace opacity',
                'consolidation','tortuous aorta','calcified nodule','lung lesion','mass/nodule (not otherwise specified)',
               'pulmonary edema/hazy opacity','costophrenic angle blunting','vascular congestion','vascular calcification',
               'linear/patchy atelectasis','elevated hemidiaphragm','pleural/parenchymal scarring','hydropneumothorax','pneumothorax',
               'enlarged hilum','multiple masses/nodules','hyperaeration','mediastinal widening','vascular redistribution',
               'mediastinal displacement','spinal degenerative changes','superior mediastinal mass/enlargement','scoliosis',
               'sub-diaphragmatic air','hernia','spinal fracture','bone lesion','increased reticular markings/ild pattern',
               'infiltration','pneumomediastinum','bronchiectasis','cyst/bullae']
labels = ['right lung', 'right upper lung', 'right mid lung', 'right lower lung', 'right hilar', 'right apical', 'right costophrenic angle',
          "right hemidiaphragm", 'left lung', 'left upper lung', 'left mid lung', 'left lower lung', 'left hilar', 'left apical',
          'left costophrenic angle', "left hemidiaphragm", "spine", 'aorta', "mediastinum", "upper mediastinum", 'cardiac silhouette',
          'left cardiac silhouette', 'right cardiac silhouette', "cavoatrial junction",
          'lung opacity', 'pleural effusion', 'atelectasis', 'pneumonia', 'enlargement of the cardiac silhouette', 'lung opacity',
          'consolidation', 'tortuosity of the thoracic aorta', 'calcification', 'lung opacity', 'granuloma', 'edema',
          'blunting of the costophrenic angle', 'vascular congestion', 'vascular calcification', 'linear atelectasis', 'pneumomediastinum',
          'contusion', 'pneumothorax', 'pneumothorax', 'hilar congestion', 'granuloma', 'emphysema', 'cardiomegaly', 'vascular congestion',
          'pleural thickening', 'scoliosis', 'cardiomegaly', 'scoliosis', 'air collection', 'hernia', 'fracture', 'fracture', 'pneumonia',
          'edema', 'pneumomediastinum', 'hypoxemia', 'pneumothorax']
entities = ['misssing finding', 'addtional finding', 'level', 'yes', 'no', 'pa view', 'ap view', 'right lung', 'right upper lung', 'right mid lung', 'right lower lung', 'right hilar', 'right apical', 'right costophrenic angle',
          "right hemidiaphragm", 'left lung', 'left upper lung', 'left mid lung', 'left lower lung', 'left hilar', 'left apical',
          'left costophrenic angle', "left hemidiaphragm", "spine", 'aorta', "mediastinum", "upper mediastinum", 'cardiac silhouette',
          'left cardiac silhouette', 'right cardiac silhouette', "cavoatrial junction",
          'lung opacity', 'pleural effusion', 'atelectasis', 'pneumonia', 'enlargement of the cardiac silhouette', 'lung opacity',
          'consolidation', 'tortuosity of the thoracic aorta', 'calcification', 'lung opacity', 'granuloma', 'edema',
          'blunting of the costophrenic angle', 'vascular congestion', 'vascular calcification', 'linear atelectasis', 'pneumomediastinum',
          'contusion', 'pneumothorax', 'pneumothorax', 'hilar congestion', 'granuloma', 'emphysema', 'cardiomegaly', 'vascular congestion',
          'pleural thickening', 'scoliosis', 'cardiomegaly', 'scoliosis', 'air collection', 'hernia', 'fracture', 'fracture', 'pneumonia',
          'edema', 'pneumomediastinum', 'hypoxemia', 'pneumothorax']
'''
lung opacity肺部浑浊, 88324
pleural effusion胸腔积液, 82474 
pneumothorax气胸, 15925
consolidation固化, 29332
fracture骨折, 9912
pleural thickening胸膜增厚, 5591
calcification钙化, 11519
pneumonia肺炎, 48599
edema水肿, 49564
vascular congestion血管充血, 23674
cardiomegaly心脏肥大, 68166
enlargement of the cardiac silhouette心脏轮廓增大, 9782
atelectasis肺不张, 92314
blunting of the costophrenic angle肋膈角变钝，胸腔积液的一种症状, 4983
heart failure心力衰竭, 6034
tortuosity of the thoracic aorta胸主动脉迂曲, 2731
hilar congestion肺门充血, 993
emphysema肺气肿, 8835
pneumomediastinum纵隔气肿, 1145
scoliosis脊柱侧弯, 2738
granuloma肉芽肿, 2401
air collection, 983
tortuosity of the descending aorta降主动脉迂曲 , 63
contusion肺挫伤, 789
gastric distention腹胀, 238
hernia疝气, 2957
hematoma血肿, 846
hypoxemia低氧血症, 136
hypertensive heart disease高血压性心脏病, 14
thymoma胸腺瘤, 7
'''

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = nn.functional.relu(self.gc1(x, adj))
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return nn.functional.log_softmax(x, dim=1)

class GraphEmbedding(nn.Module):
    def __init__(self, embedding, tokenizer):
        super(GraphEmbedding, self).__init__()
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.gcn_layer = GCN(nfeat=768, nhid=1024, nclass=768, dropout=0.5)

    def forward(self, adj):
        x = []
        with torch.no_grad():
            for label in labels:
                token = self.tokenizer(label, padding='max_length', truncation=True, max_length=10, return_tensors='pt').to(adj.device)
                output = self.embedding(token.input_ids, attention_mask=token.attention_mask, return_dict=True, mode='text')
                x.append(output.last_hidden_state[:,0,:])
        x = torch.stack(x).squeeze(dim=1)
        x = self.gcn_layer(x, adj)
        return x

