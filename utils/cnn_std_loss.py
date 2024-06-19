
from torch.nn.functional import cosine_similarity
import torch


def l1_loss(f_, f_q, ext):
    """
    param:
        f_,f_q: shape(B,C,K)
        ext: shape(B,C)
    """
    sim = 1-cosine_similarity(f_, f_q, dim=-1)
    sim = sim*ext
    return torch.sum(sim, dim=1)


def dissim1(e1, e2):
    sim = cosine_similarity(e1, e2, dim=-1)
    return torch.abs(sim)


def dissim2(e1, e2):
    sim = cosine_similarity(e1, e2, dim=-1)+1
    return torch.abs(sim)


def dissim3(e1, e2):
    sim = cosine_similarity(e1, e2, dim=-1)
    return torch.abs(torch.pow(sim, 2))


def l2_loss(f_, f_q, ext, dissim=dissim1):
    """
    param:
        f_,f_q: shape(B,C,K)
        ext: shape(B,C)
    """
    sim = (1-ext)*dissim(f_, f_q)
    return torch.sum(sim, dim=1)


def l3_loss(t, t_, dt, dt_, ext, numda_):
    dis_t = torch.pow(t-t_, 2)
    dis_dt = torch.pow(torch.sqrt(dt)-torch.sqrt(dt_), 2)
    loss = ext*(dis_t+numda_*dis_dt)
    return torch.sum(loss, dim=1)


class Loss(torch.nn.Module):
    def __init__(self, p1, p2, p3, p3_, dissim=1,) -> None:
        super().__init__()
        if dissim == 1:
            self.dissim = dissim1
        if dissim == 2:
            self.dissim = dissim2
        if dissim == 3:
            self.dissim = dissim3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p3_ = p3_

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            pred:(y,fq)
                y  -> (B,C,K+2)
                fq -> (B,C,k)
            target: (ext,t,dt)
                ext  -> (B,C)
                t,dt -> (B,C)
        """
        c = target[0].shape[-1]
        y, fq = pred
        fq = fq.unsqueeze(dim=1).repeat_interleave(c, dim=1)
        ext, t, dt = target
        f_ = y[:, :, :-2]
        t_ = y[:, :, -2]
        dt_ = y[:, :, -1]
        l1 = l1_loss(f_, fq, ext)
        l2 = l2_loss(f_, fq, ext, self.dissim)
        l3 = l3_loss(t, t_, dt, dt_, ext, self.p3_)
        total_loss=(l1*self.p1+l2*self.p2+l3*self.p3).mean()
        if torch.any(torch.isnan(total_loss)):
            print("l1",l1)
            print("l2",l2)
            print("l3",l3)
            exit()
        return total_loss


if __name__ == "__main__":
    f1 = torch.Tensor([[[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]]])
    f2 = torch.Tensor([[[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]]])
    f3 = torch.Tensor([[[0, 0.5, 0], [1, 1, 1.3], [0.5, 0.8, 1]],
                       [[0, 8, 0], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]],
                       [[1, 2, 3], [0, 0.2, 0.3], [0.5, 0.8, 1]]])
    ext = torch.Tensor([[1, 1, 1],
                       [0, 0, 0],
                       [1, 1, 0],
                       [0, 0, 1]])
    print("l1-12", l1_loss(f1, f2, ext))
    print("l1-32", l1_loss(f3, f2, ext))
    print("l2-12", l2_loss(f1, f2, ext))
    print("l2-32", l2_loss(f3, f2, ext))
    st = torch.Tensor([[[0.5, 1], [0.9, 0.4]], [[0.5, 1], [0.9, 0.4]]])
    st_ = torch.Tensor([[[0.5, 1], [0.6, 0.9]], [[0.5, 1], [0.6, 0.9]]])
    ext = torch.Tensor([[1, 0], [0, 1]])
    t = st[:, :, 0]
    dt = st[:, :, 1]
    t_ = st_[:, :, 0]
    dt_ = st_[:, :, 1]
    print(l3_loss(t, t_, dt, dt_, ext, 1))
