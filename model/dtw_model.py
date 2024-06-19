import dtw
from torch.nn.functional import cosine_similarity
import torch.nn as nn
import torch
import multiprocessing


def dist_func(x, y):
    return 1-cosine_similarity(x, y, dim=-1)


class DTW_Model(nn.Module):
    def __init__(self, w=50) -> None:
        super().__init__()
        self.w = w

    def forward(self, x1, x2):
        scores = []
        # ctx = torch.multiprocessing.get_context("spawn")
        pool = multiprocessing.Pool(10)
        for x1_, x2_ in zip(x1, x2):
            res = pool.apply_async(self.pred, args=(x1_, x2_))
            # score = self.pred(x1_, x2_)
            scores.append(res)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        scores = [s.get() for s in scores]
        scores = torch.Tensor(scores).unsqueeze(-1)
        return scores

    def pred(self, x, y):
        dist, cost, acc, path = dtw.dtw(x, y, dist_func, self.w)
        return acc[-1, -1]


class DTW(DTW_Model):
    def __init__(self, th, w=50) -> None:
        super().__init__(w)
        self.th = th

    def forward(self, x1, x2):
        scores = []
        pool = multiprocessing.Pool(20)
        for x1_, x2_ in zip(x1, x2):
            res = pool.apply_async(self.pred, args=(x1_, x2_))
            scores.append(res)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        scores = [s.get() for s in scores]
        return torch.Tensor(scores)

# model=DTW_Model()
# data=torch.randn((100,20,40))
# print(model(data,data))
