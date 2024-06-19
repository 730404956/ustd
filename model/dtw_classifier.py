import torch
from utils import dtw

def get_cost(dist, norm=True):
    """
    Calculate optimal warping path cost.
    :param x: reference sequence
    :param y: test sequence
    :param dist_func: distance function, default=euclidean distance;
    4 arguments: x, y, i, j, where x and y are sequences and i and j are indexes
    :param norm: normalization
    :return: cost, global cost matrix
    """
    rows = dist.shape[-2]
    cols = dist.shape[-1]

    cost_matrix = torch.zeros((rows + 1, cols + 1))

    cost_matrix[:, 0] = torch.inf
    cost_matrix[0, :] = torch.inf
    cost_matrix[0, 0] = 0

    for i in range(rows):
        for j in range(cols):
            cost = dist_func(x, y, i, j)
            cost_matrix[i + 1, j + 1] = cost + min(cost_matrix[i, j + 1],
                                                   cost_matrix[i + 1, j],
                                                   cost_matrix[i, j])
    if norm:
        return cost_matrix[rows, cols] / len(x), cost_matrix
    else:
        return cost_matrix[rows, cols], cost_matrix

class DTWClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bound = torch.rand(1, require_grad=True)

    def forward(self, x):
        x1, x2 = x
        dtw.get_cost()
