import torch
import torch.nn as nn
import torch.nn.functional as F


def nce_loss(pred: torch.Tensor, target: torch.Tensor):
    # B,F,T -> T,B,F
    pred = pred.permute((2, 0, 1))
    # B,F,T -> T,F,B
    target = target.permute((2, 1, 0))
    # (T,B,F)*(T,F,B) -> T,B,B
    sim_dist = pred @ target
    fz = sim_dist.diagonal(0, -2, -1)
    loss = (sim_dist - fz.unsqueeze(-1)).exp().sum(-1).log()
    return loss.mean()


def nce_loss_vanilla(pred: torch.Tensor, target: torch.Tensor):
    # B,F,T
    pred = pred
    # B,F,T -> B,T,F
    target = target.transpose(-1, -2)
    # (B,T,F)*(B,F,T) -> B,T,T
    sim_dist = target @ pred
    fz = sim_dist.diagonal(0, -2, -1)
    loss = (sim_dist - fz.unsqueeze(-1)).exp().sum(-1).log()
    return loss.mean()


class CPC_NCE_Loss(nn.Module):
    def __init__(self, t, nce_type=0) -> None:
        super().__init__()
        self.t = t
        self.nce_type = nce_type
        if nce_type == 0:
            self.nce_cal = nce_loss
        elif nce_type == 1:
            self.nce_cal = nce_loss_vanilla
        else:
            raise Exception(f"not valid type[{nce_type}]")

    def forward(self, pred: torch.Tensor, src=None):
        # B,F,T
        ht = pred[..., : -self.t]
        if src is not None:
            ht_ = src[..., self.t :]
        else:
            ht_ = pred[..., self.t :]

        return self.nce_cal(ht, ht_)

    def __str__(self) -> str:
        return f"CPC[{self.nce_type}] with t={self.t}"


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and
    label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output, target, size_average=True):
        output1, output2 = output
        distances = (output2 - output1).pow(2).sum(-1)  # squared distances
        losses = target.float() * distances + (1 - target).float() * F.relu(
            self.margin - (distances + self.eps).sqrt()
        ).pow(2)
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (
            (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        )
        negative_loss = F.relu(
            self.margin
            - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]])
            .pow(2)
            .sum(1)
            .sqrt()
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (
            (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        )  # .pow(.5)
        an_distances = (
            (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        )  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


if __name__ == "__main__":
    import time

    a = torch.randn((3, 2, 5), requires_grad=True)
    b = torch.randn((3, 2, 5))
    # a = a / a.norm(dim=-2, keepdim=True)
    # b = b / b.norm(dim=-2, keepdim=True)
    v = []
    n1 = []
    n2 = []
    for r in range(100):
        ts = time.time()
        for i in range(10000):
            nce_loss_vanilla(a, b).backward()
        v.append(time.time() - ts)
        ts = time.time()
        for i in range(10000):
            nce_loss(a, b).backward()
        n1.append(time.time() - ts)
        ts = time.time()
        for i in range(10000):
            nce_loss2(a, b).backward()
        n2.append(time.time() - ts)

    print("vanilla", torch.Tensor(v).mean())
    print("1", torch.Tensor(n1).mean())
    print("2", torch.Tensor(n2).mean())
