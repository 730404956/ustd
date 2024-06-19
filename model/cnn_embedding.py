import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_size, out_size, num_convs):
        super().__init__()
        modules = [nn.Conv2d(input_size, out_size, kernel_size=3, stride=1, padding=1), ]
        for i in range(num_convs-1):
            modules.append(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1))
        self.convs = nn.Sequential(*modules,
                                   nn.MaxPool2d(2, 2, 0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_size),
                                   )

    def forward(self, x):
        y = self.convs(x)
        return y


class CNNEmbedding(nn.Module):
    def __init__(self, f_size, f_len, hidden_size, activation="softmax", channels=1):
        super().__init__()
        assert len(hidden_size)>0, "invalid hidden size"
        self.channels = channels
        self.convs = nn.Sequential(
            ConvBlock(channels, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 256, 4),
            ConvBlock(256, 512, 4),
            ConvBlock(512, 512, 4),
        )
        fc_in_size = self.__test_fc_size__(f_size, f_len, self.channels)
        if activation == "log_softmax":
            activate = nn.LogSoftmax()
        elif activation == "relu":
            activate = nn.ReLU()
        elif activation == "prelu":
            activate = nn.PReLU()
        elif activation == "softmax":
            activate = nn.Softmax()
        # hidden_size.insert(0,fc_in_size)
        models=[nn.Flatten(),nn.Linear(fc_in_size, hidden_size[0]),]
        for i in range(1,len(hidden_size)):
            models.append(nn.ReLU(True))
            models.append(nn.Linear(hidden_size[i-1],hidden_size[i]))
        self.fc = nn.Sequential(
            *models,
            activate
        )

    def __test_fc_size__(self,  f_size, f_length, channels):
        test_input = torch.zeros((1, channels, f_size, f_length))
        cnn_output = self.convs(test_input)
        cnn_output = nn.Flatten()(cnn_output)
        out_size = cnn_output.shape[-1]
        return out_size

    def forward(self, x):
        y = self.convs(x)
        y = self.fc(y)
        return y
