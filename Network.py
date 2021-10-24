import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[offset:x.size(0)+offset, :]
        return self.dropout(x)


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM=nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1=nn.Linear(d_model*2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)


    def forward(self, x):
        res=x
        x, _ = self.LSTM(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=res+x
        return x

class ResidualGRU(nn.Module):

    def __init__(self, d_model):
        super(ResidualGRU, self).__init__()
        self.GRU=nn.GRU(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear=nn.Linear(d_model*2, d_model)


    def forward(self, x):
        res=x
        x, _ = self.GRU(x)
        x=self.linear(x)
        x=res+x
        return x


class SAKTModel(nn.Module):
    def __init__(self, n_skill, n_cat, nout, max_seq=100, embed_dim=128, pos_encode='LSTM', nlayers=2, rnnlayers=3,
    dropout=0.1, nheads=8):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        #self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        if pos_encode=='LSTM':
            self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU':
            self.pos_encoder = nn.ModuleList([ResidualGRU(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU2':
            self.pos_encoder = nn.GRU(embed_dim,embed_dim, num_layers=2,dropout=dropout)
        elif pos_encode=='RNN':
            self.pos_encoder = nn.RNN(embed_dim,embed_dim,num_layers=2,dropout=dropout)
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(n_skill, embed_dim)
        self.cat_embedding = nn.Embedding(n_cat, embed_dim, padding_idx=0)
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout) for i in range(nlayers)]
        conv_layers = [nn.Conv1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        deconv_layers = [nn.ConvTranspose1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        layer_norm_layers = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        layer_norm_layers2 = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, nout)
        self.downsample = nn.Linear(embed_dim*2,embed_dim)

    def forward(self, numerical_features, categorical_features=None):
        device = numerical_features.device
        numerical_features=self.embedding(numerical_features)

        #categorical_features=self.cat_embedding(categorical_features).sum(-2)
        x = numerical_features#+categorical_features



        x = x.permute(1, 0, 2)

        #res = x

        #x, _ = self.pos_encoder(x)

        for lstm in self.pos_encoder:
            x=lstm(x)

        #x = self.downsample(x)
        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)

        #x = res+x

        #att_mask = future_mask(x.size(0)).to(device)
        #att_mask = att_mask.expand(self.nheads, *att_mask.shape).transpose(1,0).reshape(-1,*att_mask.shape[1:])
        #x = self.transformer_encoder(x)

        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                               self.deconv_layers):
            #LXBXC to BXCXL
            res=x
            x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm1(x)
            x=transformer_layer(x)
            x=F.relu(deconv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm2(x)
            x=res+x

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
