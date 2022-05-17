import torch

from torch import Tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_zeros_tensor(size: int) -> Tensor:
    """
        Create a Tensor with zeros of a defined size and mounted in the available device.
    :param size: int
        Size of the tensor.
    :return:
        Tensor
    """
    return torch.zeros(size).to(device)


def create_ones_tensor(size: int) -> Tensor:
    """
        Create a Tensor with ones of a defined size and mounted in the available device.
    :param size: int
        Size of the tensor.
    :return:
        Tensor
    """
    return torch.ones(size).to(device)
