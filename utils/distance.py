import torch.nn as nn
import torch


class MatrixCosineDistance(nn.Module):
    def __init__(self, f_dim, t_dim) -> None:
        super().__init__()
        self.f_dim = f_dim
        self.t_dim = t_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask=None):
        n1 = x1.norm(dim=self.f_dim, keepdim=True)
        n2 = x2.norm(dim=self.f_dim, keepdim=True)
        if self.f_dim < self.t_dim:
            norm = torch.matmul(n1.transpose(self.t_dim, self.f_dim), n2)
            prod = torch.matmul(x1.transpose(self.t_dim, self.f_dim), x2)
        else:
            norm = torch.matmul(n1, n2.transpose(self.t_dim, self.f_dim))
            prod = torch.matmul(x1, x2.transpose(self.t_dim, self.f_dim))
        dist = prod / norm
        if mask is not None:
            mask_tensor = torch.ones_like(dist)
            for i, (w, h) in enumerate(zip(mask[0], mask[1])):
                mask_tensor[i, w:] = mask[2]
                mask_tensor[i, :, h:] = mask[2]
            dist = dist * mask_tensor
        return dist


if __name__ == "__main__":
    dis = MatrixCosineDistance(-2, -1)
    x1 = torch.ones((3, 1, 4, 11))
    x2 = torch.ones((3, 1, 4, 14))
    print(dis(x1, x2).shape)
