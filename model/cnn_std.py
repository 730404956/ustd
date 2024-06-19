import torch
import torch.nn as nn
from model.cnn_embedding import CNNEmbedding
from model.LSTM import SimpleLSTM
from utils.seq_util import combine


class CNN_STD(nn.Module):
    def __init__(self, f_size, f_length, p_size, p_length, k, c, channels=1, out_size=None, cnn_load=None, ebd_load=None) -> None:
        super().__init__()
        self.c = c
        self.k = k
        self.embedding_func = SimpleLSTM(p_size, p_length, k)
        self.cnn = CNNEmbedding(f_size, f_length, [k], "relu", channels)
        if cnn_load:
            stat = torch.load(cnn_load, map_location="cpu")
            stat = {k: v for k, v in stat.items() if (k in stat and 'fc' not in k)}
            stat_src = self.cnn.state_dict()
            stat_src.update(stat)
            self.cnn.load_state_dict(stat_src)
        if ebd_load:
            stat = torch.load(ebd_load, map_location="cpu")
            self.embedding_func.load_state_dict(stat)
        self.flatten = nn.Flatten()
        self.in_size = self.__test_fc_size__(f_size, f_length, p_size, p_length, channels)
        self.fc = nn.Sequential(
            nn.Linear(self.in_size, out_size or k+2),
            nn.ReLU()
        )

    def __test_fc_size__(self,  f_size, f_length, p_size, p_length, channels):
        test_input = torch.zeros((self.c, channels, f_size, f_length))
        p = torch.zeros((self.c, p_length, p_size))
        cnn_output = self.cnn(test_input)
        cnn_output = self.flatten(cnn_output)
        lstm_output = self.embedding_func(p)
        fc_in = torch.cat([cnn_output, lstm_output], dim=-1)
        out_size = torch.flatten(fc_in, start_dim=1).shape[-1]
        return out_size

    def forward(self, input):
        """
            input: tuple(x,p)
        """
        x, p = input
        fq = self.embedding_func(p)
        y = self.cnn(x)
        y = self.flatten(y)
        fc_in = torch.cat([y, fq.repeat_interleave(self.c, 0)], -1)
        y = self.fc(fc_in)
        if self.c > 1:
            y = combine(y, self.c)
        return (y, fq)


if __name__ == "__main__":
    c = 3
    k = 1024
    f_size = 40
    f_length = 800
    p_size = 50
    p_length = 50
    batch_size = 16
    model = CNN_STD(f_size, f_length, p_size, p_length, k, c)
    # model.pretrain_on_gsc(35)
    batch_data = torch.rand([batch_size, 1, f_size, f_length])
    p = torch.rand([batch_size, p_length, p_size])
    y = model((batch_data, p))
    print(y.shape)
