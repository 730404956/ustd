import torch
import torch.nn as nn


class CNN_STD(nn.Module):
    def __init__(self, ebd1, ebd2, matrix_gen, in_channel=1) -> None:
        super().__init__()
        self.ebd1 = ebd1
        self.ebd2 = ebd2
        self.matrix_gen = matrix_gen
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 1, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )

    def init_fc(self, x):
        x1,x2=x
        out1 = self.ebd1(x1)
        out2 = self.ebd2(x2)
        out_m = self.matrix_gen(out1, out2)
        cnn_out = self.cnn(out_m)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out.shape[-1], 2),
            nn.LogSoftmax()
        )
        return self

    def forward(self, input):
        """
            input: tuple(x1,x2)
        """
        x1, x2 = input
        out1 = self.ebd1(x1)
        out2 = self.ebd2(x2)
        out_m = self.matrix_gen(out1, out2)
        cnn_out = self.cnn(out_m)
        y = self.fc(cnn_out)
        return y


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
