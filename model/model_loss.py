from torch import nn
from model.solver import Solver


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model = None

    def init_model(self, args):
        self.model = Solver(**vars(args))

    def forward(self, inputs, target=None):
        NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, anchor, utimate_selected_sents = self.model(
            *inputs)
        return NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, anchor, utimate_selected_sents
