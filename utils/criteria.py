import torch

THELHOLD = 0.5


def check_dim(pred: torch.Tensor):
    pred = pred.squeeze()
    if len(pred.shape) > 1:
        pred = pred.argmax(-1)
    else:
        bg = pred > THELHOLD
        sm = pred <= THELHOLD
        pred[bg] = 1
        pred[sm] = 0
    return pred


def TP(pred_res: torch.Tensor, target: torch.Tensor):
    return (target[pred_res == 1] == 1).sum()


def FP(pred_res: torch.Tensor, target: torch.Tensor):
    return (target[pred_res == 1] == 0).sum()


def TN(pred_res: torch.Tensor, target: torch.Tensor):
    return (target[pred_res == 0] == 0).sum()


def FN(pred_res: torch.Tensor, target: torch.Tensor):
    return (target[pred_res == 0] == 1).sum()


def accuracy(pred: torch.Tensor, target: torch.Tensor):
    pred = check_dim(pred)
    acc = (pred == target).float().mean()
    return acc


def accuracy_thelhold(pred: torch.Tensor, target: torch.Tensor):
    big = pred > 0.5
    small = pred <= 0.5
    pred[big] = 1
    pred[small] = 0
    return (pred == target).float().mean()


def ap(pred: torch.Tensor, target: torch.Tensor):
    pred = check_dim(pred)
    tp = TP(pred, target)
    if tp == 0:
        return 0
    fp = FP(pred, target)
    ap = tp/(tp+fp)
    return ap


def recall(pred: torch.Tensor, target: torch.Tensor):
    pred = check_dim(pred)
    tp = TP(pred, target)
    if tp == 0:
        return 0
    fn = FN(pred, target)
    ap = tp/(tp+fn)
    return ap


def FalseAlarm(pred: torch.Tensor, target: torch.Tensor):
    pred = check_dim(pred)
    fp = FP(pred, target)
    if fp == 0:
        return 0
    tp = TP(pred, target)
    fp = fp/(tp+fp)
    return fp


def empty_criteria(x, y):
    return 0


def Cmax(pred: torch.Tensor, ext: torch.Tensor):
    p_h1 = ext.sum()/ext.shape[0]
    p_h0 = (1-ext).sum()/ext.shape[0]
    ll_rt = torch.log(p_h0/p_h1)


class Recoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.results = []

    def forward(self, out, gt):
        self.results.append()
        return 0

    def save_results(self, path):
        with open(path) as f:
            f.write()


class TWV(torch.nn.Module):
    def __init__(self, beta) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, ext: torch.Tensor):
        pred_res = pred.argmax(-1)
        n_fr = FN(pred_res, ext)
        n_ref = max(1, (ext == 1).sum())
        n_nt = max(1, (ext == 0).sum())
        n_fa = FP(pred_res, ext)
        p_fr = n_fr/n_ref
        p_fa = n_fa/n_nt
        return 1-p_fr-p_fa*self.beta


if __name__ == "__main__":
    pred = torch.ones(10)
    tgt = torch.zeros(10)
    print(TP(pred, tgt))
    print(TN(pred, tgt))
    print(FP(pred, tgt))
    print(FN(pred, tgt))
    print(ap(pred, tgt))
    print(recall(pred, tgt))
