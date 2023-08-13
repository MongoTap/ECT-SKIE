from torch import nn
from model.gumbel_softmax import gumbel_softmax_topK, gumbel_softmax, no_gumbel_select
from model.infonce import InfoNCE
from model.uniform_loss import UniformLoss
from model.module import Encoder, ImportanceScore
import torch.nn.functional as F
from utils import *


class Solver(nn.Module):

    def __init__(self,
                 embedding_size=None,
                 hidden_size=None,
                 hidden_layers=None,
                 dropout=None,
                 in_LN=True,
                 hid_LN=True,
                 out_LN=True,
                 softmax_tau=1.,
                 predict=False):
        super().__init__()
        self.emb_size = embedding_size
        self.hid_size = hidden_size
        self.hid_layers = hidden_layers
        self.dropout = dropout
        self.in_LN = in_LN
        self.hid_LN = hid_LN
        self.out_LN = out_LN
        self.predict = predict
        self.ImportanceScore = ImportanceScore(self.emb_size * 2, self.hid_size, self.hid_layers, 1,
                                               self.dropout, self.in_LN, self.hid_LN, self.out_LN)
        self.sent_linear = nn.Linear(self.emb_size, self.emb_size)
        self.trans_encoder = Encoder(self.emb_size)
        self.qa_encoder = Encoder(self.emb_size)
        self.temp = softmax_tau
        self.infonce = InfoNCE()
        self.container_loss = InfoNCE()
        self.container_loss_v2 = InfoNCE(negative_mode='paired')
        # self.redundancy_InfoNCE = InfoNCE(negative_mode='paired')
        self.uniform_loss = UniformLoss()
        self.KLDivLoss = nn.KLDivLoss(reduction='sum')
        self.K_slot = 200
        # self.top_K = 20
        self.top_K_lower_bound = 5
        self.compression_rate = 0.15
        self.container = nn.Parameter(torch.randn(self.K_slot, self.emb_size), requires_grad=True)
        self.init_container()
        # self.convert2d = Conver2d(input_dim=self.emb_size, dropout=0.05)

    def init_container(self):
        nn.init.ones_(self.container)

    def forward(self, padded_trans, trans_len, mean_qa, mask, att_mask):
        # container = F.normalize(self.container, dim=-1)
        trans = self.sent_linear(padded_trans)
        sim_score = torch.matmul(self.container, trans.permute(0, 2, 1))  # [B, K, N]
        mask_expand = mask.unsqueeze(dim=1).expand(-1, self.K_slot, -1)  # [B, K, N]
        sim_score = sim_score.masked_fill(mask_expand, -np.inf)

        if self.predict:
            selected_position = no_gumbel_select(logits=sim_score, tau=self.temp, hard=True,
                                                 dim=-1)  # [B, K, N]
        else:
            selected_position, selected_score_soft = gumbel_softmax(logits=sim_score,
                                                                    tau=self.temp,
                                                                    hard=True,
                                                                    dim=-1)  # [B, K, N]

        # Redundancyloss = self.bolster_repeat_select(selected_score_soft)

        selected_sent = torch.matmul(selected_position, trans)  # [B,K,N] * [B, N, d] =  [B, K, d]

        expand_container = self.container.unsqueeze(dim=0).expand(padded_trans.size(0), -1, -1)
        joint_rep = torch.cat([expand_container, selected_sent], dim=-1)

        out = self.ImportanceScore(joint_rep)  # [B, K, 1]
        out = out.squeeze()  # [B, K]

        ConciseLoss = self.ConsiseControl(logits=out)

        mean_selected_sent, discrete_mask, select_length = self.selectByRate(out=out,
                                                                             padded_trans=selected_sent,
                                                                             trans_len=trans_len)

        # two MLPs for trans-sent and qa-sent
        anchor = self.trans_encoder(mean_selected_sent)
        positive = self.qa_encoder(mean_qa)

        NCEloss = self.infonce(anchor, positive)

        # y_pred = self.predictor(mean_selected_sent) if self.predict else None

        # with torch.no_grad():
        Redundancyloss = self.Redundancy(selected_sent=selected_sent, container=self.container)
        # Redundancyloss = 0.

        # Uniformloss = self.uniform_loss(self.convert2d(self.container))

        with torch.no_grad():
            selected_trans = torch.matmul(selected_position, padded_trans)
            # Uniformloss = self.compute_uniformity(discrete_mask=discrete_mask, trans=selected_trans)
            Uniformloss = self.uniform_loss(self.container)
            # Uniformloss = self.uniform_loss()
            # [B, 1, K] * [B, K, N] = [B, 1, N]
            res_mask = torch.matmul(discrete_mask.unsqueeze(dim=1), selected_position).squeeze()  # [B, N]

            utimate_selected_sents = self.get_utimate_sents(discrete_mask=discrete_mask, trans=selected_trans)
            # print(res_mask.size())
            # print(discrete_mask.size())
            # print(selected_position.size())

        return NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, mean_selected_sent, utimate_selected_sents

    # def bolster_repeat_select(self, slot_select_res):
    #     slot_select_dist = F.log_softmax(slot_select_res.sum(dim=1), dim=-1)  # [B, N]
    #     prior_dist = (torch.ones_like(slot_select_dist) / slot_select_dist.size(-1)).detach()
    #     return self.KLDivLoss(slot_select_dist, prior_dist) / prior_dist.size(0)

    # compute the uniformity between selected sentences
    def compute_uniformity(self, discrete_mask, trans):
        uniform_list = []
        for each_mask, tran in zip(discrete_mask, trans):
            one_indices = torch.nonzero(each_mask).squeeze()
            row_indices = torch.arange(one_indices.size(0))
            new_mask = torch.zeros(one_indices.size(0), each_mask.size(0)).to(trans.device)
            new_mask[row_indices, one_indices] = 1.
            uniform_list.append(self.uniform_loss(torch.matmul(new_mask, tran)))

        return torch.as_tensor(uniform_list).mean()

    def get_utimate_sents(self, discrete_mask, trans):
        sents_list = []
        for each_mask, tran in zip(discrete_mask, trans):
            one_indices = torch.nonzero(each_mask).squeeze()
            row_indices = torch.arange(one_indices.size(0))
            new_mask = torch.zeros(one_indices.size(0), each_mask.size(0)).to(trans.device)
            new_mask[row_indices, one_indices] = 1.
            sents_list.append(torch.matmul(new_mask, tran))
        return sents_list

    # # gumbel softmax sample topK sentences.
    # def selectByTopK(self, out, topK, padded_trans, temperature):
    #     # out = out.masked_fill(mask, -np.inf)
    #     select_res = gumbel_softmax_topK(out, top_k=topK, tau=temperature, hard=True)
    #     select_res = select_res.unsqueeze(dim=-1)  # [bsz, padded_sent_len, 1]

    #     return (padded_trans * select_res).sum(dim=1) / self.top_K

    def ConsiseControl(self, logits):
        p_prior = (torch.ones_like(logits) / logits.size(-1)).detach()
        log_p_i = F.log_softmax(logits, dim=-1)
        return self.KLDivLoss(log_p_i, p_prior) / logits.size(0)

    def Redundancy(self, selected_sent, container):
        # # Version 1: bernouli sample from the selected_sent, divided into two parts as anchor and positive.
        # Batchsize = selected_sent.size(0)
        # selected_sent_T = selected_sent.permute(1, 0, 2)  # [K, B, d]
        # anchor_list = []
        # positive_list = []
        # for slot in selected_sent_T:
        #     first_index, second_index = bernouli_sample(Batchsize)
        #     anchor_list.append(slot[first_index].mean(dim=0))
        #     positive_list.append(slot[second_index].mean(dim=0))
        # anchor = torch.stack(anchor_list)
        # positive = torch.stack(positive_list)
        # return self.container_loss(anchor, positive)

        # Version 2: container as anchor, selected_sent as positive, other slots as negative.
        mean_selected_sent = selected_sent.mean(dim=0)  # [K, d]
        negative_list = []
        for row_exclude in range(container.size(0)):
            negative_list.append(torch.cat((container[:row_exclude], container[row_exclude + 1:])))

        negative = torch.stack(negative_list)  # [K, K-1, d]

        return self.container_loss_v2(container, mean_selected_sent, negative)

        # gumbel softmax sample sentences by a certain rate.
    def selectByRate(self, out, padded_trans, trans_len, temperature=1.):
        select_res_list = []
        select_length = (torch.clamp(torch.ceil(torch.as_tensor(trans_len) * self.compression_rate).int() +
                                     self.top_K_lower_bound,
                                     max=self.K_slot)).detach().to(out.device)
        for each, select_len in zip(out, select_length):
            select_res_list.append(
                gumbel_softmax_topK(each, top_k=select_len.item(), tau=temperature, hard=True))
        discrete_mask = torch.stack(select_res_list, dim=0)  # [B, K]

        # discrete_mask_weighted for example:
        # [ [0.33, 0.33, 0.33, 0, 0],
        #   [0.25, 0.25, 0.25, 0.25, 0],
        #   [0.5, 0.5, 0, 0, 0],
        #   [1., 0, 0, 0, 0] ]
        discrete_mask_weighted = (discrete_mask / select_length.view(discrete_mask.size(0), 1))

        # [bsz, padded_sent_len, d] * [bsz, padded_sent_len, 1]
        return (padded_trans *
                discrete_mask_weighted.unsqueeze(dim=-1)).sum(dim=1), discrete_mask, select_length


    # return the index of two parts sampled.
def bernouli_sample(seq_len):
    prob = torch.ones(seq_len) / 2  # probability = 0.5
    first_part = torch.bernoulli(prob)
    first_index = torch.nonzero(first_part == 1.).squeeze()

    sampled_length = len(first_index)
    if sampled_length < (seq_len / 4) or sampled_length > ((seq_len * 3) / 4):
        return bernouli_sample(seq_len)

    second_index = torch.nonzero(first_part == 0.).squeeze()

    return first_index, second_index
