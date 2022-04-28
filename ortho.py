# Make your model Dynamical Isometric!
# Mainly inspired by https://arxiv.org/pdf/1806.05393.pdf
#       and a bit by https://arxiv.org/pdf/2203.05483.pdf
import torch
from torch import nn


def orthogonalize(w):
    """
    from https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
    by Zing Lee
    Orthogonalize square/non-square matrix 'w' using svd.
    :param w: weight tensor
    :return: orthogonal weight tensor
    """
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    mat = u @ vh
    return mat


def to_matrix(w):
    return w.reshape((w.size(0), torch.prod(torch.tensor(w.size()[1:])).item()))


def to_tensor(w, size):
    return w.reshape(size)


def orthogonlize_model(module: nn.Module):
    """
    Goes over all the weights and orthogonlize them in-place using svd.
    :param module: pytorch module or a model containing modules.
    """
    for param in module.parameters():
        w = param.data
        if len(w.size()) > 1:
            w = orthogonalize(to_matrix(w))
            param.data = to_tensor(w, param.data.size())
