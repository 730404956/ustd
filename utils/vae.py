import torch
import torch.nn.functional as F


def kl_loss(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))


def recon_loss(recon, origin):
    return F.mse_loss(recon, origin)


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = eps * std + mu
    return z
