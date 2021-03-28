import torch


def calculate_accuracy(prediction: torch.Tensor, target: torch.Tensor, device) -> (float,float,torch.Tensor):
    """
    Returns the BER and FER matching between the two tensors
    Also the errors vec
    :param prediction: decoded word
    :param target: target word
    :param device: device
    :return: BER,FER
    """
    # Accuracy
    prediction = prediction.long()
    target = target.long()
    ber_acc = torch.mean(torch.eq(prediction, target).float()).item()
    all_bits_sum_vector = torch.sum(torch.abs(prediction - target), 1).long()
    fer_acc = torch.eq(all_bits_sum_vector, torch.LongTensor(1).fill_(0).to(device=device)).float().mean().item()
    return max([1 - ber_acc, 0.0]), max([1 - fer_acc, 0.0]), all_bits_sum_vector.nonzero().reshape(-1)
