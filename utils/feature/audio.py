from torchaudio.transforms import MFCC as torch_mfcc
from torchaudio.transforms import MelSpectrogram as mel
from torchaudio.transforms import ComputeDeltas as Delta
import torch


class Feature(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000, n_feat: int = 40) -> None:
        """
        out : C,F,T
        """
        super().__init__()
        self.sr = sample_rate
        self.n_feat = n_feat

    def signal_norm(self, signal):
        # signal[B,T]
        signal = signal / (signal.abs().max(dim=-1,keepdim=True)[0])
        return signal
    
    def forward(self,x):
        return self.signal_norm(x)


class Mel_Spec(Feature):
    def __init__(self, log=False, sample_rate: int = 16000, n_feat: int = 40, win_length=None, hop_length=None) -> None:
        super().__init__(sample_rate, n_feat)
        self.mel_spec = mel(sample_rate, win_length, win_length, hop_length,n_mels=n_feat)
        self.log=log

    def forward(self, signal):
        signal = self.signal_norm(signal)
        feat = self.mel_spec(signal)
        if self.log:
            feat=torch.log(feat+1e-8)
        return feat


class MFCC(Feature):

    def __init__(self, delta=0,  n_feat: int = 40, sample_rate: int = 16000, win_length=None, hop_length=None, delta_opt="cat") -> None:
        super().__init__(sample_rate, n_feat)
        melkwgs = {
            "win_length": win_length, "hop_length": hop_length,"n_fft":max(400,win_length)
        }
        self.mfcc = torch_mfcc(self.sr, self.n_feat, melkwargs=melkwgs)
        self.delta_opt = delta_opt
        self.delta = Delta()
        self.delta_orders = delta

    def forward(self, signal):
        signal = self.signal_norm(signal)
        feature = self.mfcc(signal)
        deltas = [feature]
        orders = self.delta_orders
        while orders > 0:
            deltas.append(self.delta(deltas[-1]))
            orders -= 1
        if self.delta_opt == "cat":
            feature = torch.cat(deltas, dim=-2)
        elif self.delta_opt == "add":
            for f in deltas[1:]:
                feature += f
        return feature
