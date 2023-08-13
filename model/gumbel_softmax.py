import torch
from torch.nn.functional import gumbel_softmax as torch_gumbel

Tensor = torch.Tensor


def gumbel_softmax_topK(logits: Tensor,
                        top_k=10,
                        tau: float = 1,
                        hard: bool = False,
                        dim: int = -1) -> Tensor:

    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
               )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        # print(y_soft)
        # index = y_soft.max(dim, keepdim=True)[1]
        _, index = y_soft.topk(top_k, dim=-1)
        # print(index)
        y_hard = torch.zeros_like(logits,
                                  memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_softmax_by_N_times(logits: Tensor,
                              tau: float = 1,
                              hard: bool = False,
                              times_n: int = 1,
                              dim: int = -1) -> Tensor:
    if times_n < 1:
        AssertionError('sample one time at least.')
    # sample for the first
    sampled_y = torch_gumbel(logits=logits, tau=tau, hard=hard, dim=dim)

    # sample for the other times (times_n - 1)
    # There is a probability that the same position is sampled twice. Hence, exist the element value > 1 in the sampled_y matrix.
    for i in range(times_n - 1):
        sampled_y = sampled_y + torch_gumbel(logits=logits, tau=tau, hard=hard, dim=dim)

    return sampled_y


def gumbel_softmax(logits: Tensor,
                   tau: float = 1,
                   hard: bool = False,
                   eps: float = 1e-10,
                   dim: int = -1) -> Tensor:

    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
               )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits,
                                  memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, y_soft


def no_gumbel_select(logits: Tensor,
                     tau: float = 1,
                     hard: bool = False,
                     eps: float = 1e-10,
                     dim: int = -1) -> Tensor:

    y_soft = (logits / tau).softmax(dim)
    # print(y_soft.size())

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits,
                                  memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret