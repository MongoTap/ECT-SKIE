import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sequential import Sequential


class Encoder(nn.Module):

    def __init__(self, hidden_size, dropout=0.4):
        super(Encoder, self).__init__()
        self.h_dim = hidden_size
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)  # the first ff
        self.LN = nn.LayerNorm(self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)  # mu
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc2(self.dropout1(self.activation(self.fc1(self.LN(x)))))
        return self.dropout2(x)


class ImportanceScore(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_size,
                 hidden_layers,
                 out_dim,
                 dropout,
                 in_LN=True,
                 hid_LN=True,
                 out_LN=True,
                 out=True):
        super(ImportanceScore, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout
        self.feedforward = self.ff(in_size=input_dim,
                                   hidden_size=hidden_size,
                                   hidden_layers=hidden_layers,
                                   out_dim=out_dim,
                                   in_LN=in_LN,
                                   hid_LN=hid_LN,
                                   out_LN=out_LN,
                                   out=out)

    def get_block(self, in_size, hidden_size, LN, act=True, drop=True):
        result = nn.Sequential(
            nn.LayerNorm(in_size) if LN else None,
            nn.Dropout(p=self.dropout) if drop else None, nn.Linear(in_size, hidden_size),
            nn.ReLU() if act else None)

        for layer in result.named_modules():
            self.weights_init(layer)

        return result

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

    def ff(self,
           in_size,
           hidden_size,
           hidden_layers,
           out_dim,
           in_LN=True,
           hid_LN=True,
           out_LN=True,
           out=True):

        ff_seq = list()
        ff_seq.extend(self.get_block(in_size, hidden_size[0], LN=in_LN))
        for i in range(1, hidden_layers):
            ff_seq.extend(self.get_block(hidden_size[i - 1], hidden_size[i], LN=hid_LN))
        if out:
            ff_seq.extend(self.get_block(hidden_size[-1], out_dim, LN=out_LN, act=False, drop=False))

        return Sequential(*ff_seq)

    def forward(self, x):
        x = self.sigmoid(self.feedforward(x))
        # x = self.feedforward(x)
        return x


class Conver2d(nn.Module):

    def __init__(self, input_dim, dropout):
        super(Conver2d, self).__init__()
        self.dropout = dropout
        self.feedforward = self.ff(in_size=input_dim)

    def get_block(self, in_size, hidden_size, LN=False, act=True, drop=True):
        result = nn.Sequential(
            nn.BatchNorm1d(in_size) if LN else None,
            nn.Dropout(p=self.dropout) if drop else None, nn.Linear(in_size, hidden_size),
            nn.ReLU() if act else None)

        for layer in result.named_modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))

        return result

    def ff(self, in_size):
        ff_seq = list()
        ff_seq.extend(self.get_block(in_size=in_size, hidden_size=150, act=True))
        ff_seq.extend(self.get_block(in_size=150, hidden_size=2, act=False))

        return Sequential(*ff_seq)

    def forward(self, x):
        return F.normalize(self.feedforward(x), dim=-1)


class RiskPredictor(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_size,
                 hidden_layers,
                 dropout,
                 in_LN=True,
                 hid_LN=True,
                 out_LN=True,
                 out=True):
        super(RiskPredictor, self).__init__()
        self.dropout = dropout
        self.feedforward = self.ff(in_size=input_dim,
                                   hidden_size=hidden_size,
                                   hidden_layers=hidden_layers,
                                   in_LN=in_LN,
                                   hid_LN=hid_LN,
                                   out_LN=out_LN,
                                   out=out)
        self.encoder = self.ff(in_size=input_dim,
                               hidden_size=[input_dim, input_dim, input_dim],
                               hidden_layers=3,
                               in_LN=False,
                               hid_LN=False,
                               out_LN=False,
                               out=False)

    def get_block(self, in_size, hidden_size, LN, act=True, drop=True):
        result = nn.Sequential(
            nn.BatchNorm1d(in_size) if LN else None,
            nn.Dropout(p=self.dropout) if drop else None, nn.Linear(in_size, hidden_size),
            nn.ReLU() if act else None)

        for layer in result.named_modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))

        return result

    def ff(self, in_size, hidden_size, hidden_layers, in_LN=True, hid_LN=True, out_LN=True, out=True):

        ff_seq = list()
        ff_seq.extend(self.get_block(in_size, hidden_size[0], LN=in_LN))
        for i in range(1, hidden_layers):
            ff_seq.extend(self.get_block(hidden_size[i - 1], hidden_size[i], LN=hid_LN))
        if out:
            ff_seq.extend(self.get_block(hidden_size[-1], 1, LN=out_LN, act=False, drop=False))

        return Sequential(*ff_seq)

    def filter(self, x):
        x_len = []
        for sample in x:
            x_len.append(len(sample))
        sents_rep = torch.cat(x, dim=0)
        encoded_rep = self.encoder(sents_rep)
        res = []
        for sample in encoded_rep.split(x_len, dim=0):
            res.append(sample.mean(dim=0))
        return torch.stack(res, dim=0)

    def forward(self, x):
        return self.feedforward(x)