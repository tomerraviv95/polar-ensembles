from python_code.utils.python_utils import llr_to_bits
from globals import DEVICE
import torch


def generator_criterion(x, u, code_gm, not_satisfied):
    """
    Stops if x=Gu
    """
    x_est = torch.matmul(u, code_gm) % 2
    not_equal = ~torch.all(x == x_est, dim=1)
    new_indices = not_satisfied[not_equal]
    return new_indices


def hard_decision_condition(satisfied, llr_words):
    words = llr_to_bits(llr_words).double()
    equal_flag = ~torch.eq(torch.sum(words, dim=1), torch.DoubleTensor(1).fill_(0).to(device=DEVICE))
    new_indices = satisfied[equal_flag]
    return new_indices


def syndrome_condition(unsatisfied, llr_words, code_pcm):
    words = llr_to_bits(llr_words).float()
    syndrome = torch.fmod(torch.mm(words, torch.tensor(code_pcm.T).float().to(device=DEVICE)), 2)
    equal_flag = ~torch.eq(torch.sum(torch.abs(syndrome), dim=1), torch.FloatTensor(1).fill_(0).to(device=DEVICE))
    new_unsatisfied = unsatisfied[equal_flag]
    return new_unsatisfied
