from python_code.crc_codes.crc import encode_by_generator_matrix
import torch


def generator_criterion(x, u, factor_graph, not_satisfied):
    """
    Stops if x=Gu
    """
    x_est = torch.matmul(u, factor_graph) % 2
    not_equal = ~torch.all(x == x_est, dim=1)
    new_indices = not_satisfied[not_equal]
    return new_indices


def crc_criterion(u, info_ind, crc_ind, poly, crc_gm, not_satisfied):
    """
    Stops if crc of current decoded word is in the CRC code
    """
    crc_bits = u[:, crc_ind]
    message_bits = u[:, info_ind]
    current_remainder = encode_by_generator_matrix(message_bits, poly, crc_gm)
    correct = torch.all(crc_bits == current_remainder, dim=1)
    new_indices = not_satisfied[~correct]
    return new_indices
