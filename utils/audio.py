import torchaudio
from torchaudio.functional import resample, add_noise
import random
from torch import Tensor
import librosa
import torch
from pathlib import Path


def random_aug(waveform, sr):
    effects = [
        ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
        ["speed", "0.8"],  # reduce the speed
        # This only changes sample rate, so it is necessary to
        # add `rate` effect with original sample rate after this.
        ["rate", f"{sr}"],
        ["reverb", "-w"],  # Reverbration gives some dramatic feeling
    ]

    # Apply effects
    waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)


class NoiseAdder(torch.nn.Module):
    def __init__(self, noise_path, snr) -> None:
        super().__init__()
        self.snr = snr
        self.noise_audios = list_audio(noise_path)

    def get_snr(self, batch_size):
        if isinstance(self.snr, tuple):
            snr = torch.randint(self.snr[0], self.snr[1], (batch_size,))
        elif isinstance(self.snr, (int, float)):
            snr = torch.full((batch_size,), self.snr)
        else:
            raise Exception(f"snr not support!({self.snr})")
        return snr

    def forward(self, waveform: Tensor):
        noise_path = random.choice(self.noise_audios)
        batch_size = waveform.size(0)
        noise, _ = load_audio(noise_path)
        seg_noise = []
        for i in range(batch_size):
            noi, _ = random_seg(noise, (waveform.size(-1), waveform.size(-1)))
            seg_noise.append(noi.squeeze())
        noise = torch.stack(seg_noise).to(waveform.device)
        snr = self.get_snr(batch_size).to(waveform.device)
        noisy_waveform = add_noise(waveform, noise, snr)
        return noisy_waveform


def signal_norm(signal):
    signal = signal / (signal.abs().max(dim=-1, keepdim=True)[0])
    return signal


def list_audio(*root):
    results = []
    for rt in root:
        audios = list(Path(rt).rglob("*.[wf][al][va]*"))
        results.extend(audios)
    return results


def remove_silent(waveform: Tensor, sr=16000, interval=2):
    current_frame = 0
    wavs = []
    start_frame = 0
    while current_frame < waveform.size(-1):
        sub_wav = waveform[..., current_frame : current_frame + sr * interval]
        current_frame += sr * interval
        sub_wav, bias = librosa.effects.trim(sub_wav, top_db=30, frame_length=256, hop_length=64)
        if sub_wav.size(-1) > 0:
            wavs.append(sub_wav)
            if start_frame == 0:
                start_frame = bias[0]
    waveform = torch.concat(wavs, -1)
    return waveform, start_frame


def load_audio(audio_path, tgt_sr=None):
    """load audio using torchaudio

    Args:
        audio_path (str): audio apth
        tgt_sr (int, optional): set target sample rate. Defaults to None.

    Returns:
        wavform: Tensor
        length: length in seconds
    """
    waveform, sr = torchaudio.load(audio_path)
    waveform = signal_norm(waveform)
    length = waveform.size(-1) / sr
    if tgt_sr is not None and sr != tgt_sr:
        waveform = resample(waveform, sr, tgt_sr)
    return waveform, length


def randrange(start, stop):
    return int(random.random() * (stop - start) + start)


def random_seg(seq: Tensor, seg_len: tuple, dim=-1, keep=None):
    min_len, max_len = seg_len
    src_seq_len = seq.size(dim)
    if src_seq_len > max_len:
        if keep is None:
            random_len = randrange(min_len, max_len)
            t_start = randrange(0, src_seq_len - random_len)
        else:
            keep_len = keep[1] - keep[0]
            min_len = max(min_len, keep_len)
            max_len = max(min(max_len, src_seq_len - keep[0]), min_len)
            random_len = randrange(min_len, max_len)
            if src_seq_len - random_len < keep[0]:
                t_start = src_seq_len - random_len
            elif keep[0] == 0:
                t_start = 0
            else:
                t_start = randrange(max(0, keep[1] - random_len), keep[0])
        t_end = t_start + random_len
        if dim != -1:
            seq = seq.transpose(-1, dim)[..., t_start:t_end].transpose(-1, dim)
        else:
            seq = seq[..., t_start:t_end]
    else:
        t_start = 0
        t_end = src_seq_len
    if keep is not None:
        return seq, (keep[0] - t_start, keep[1] - t_start)
    else:
        return seq, (t_start, t_end)


def random_drop(seq: Tensor, drop_length, dim=-1):
    start_frame = random.randrange(0, seq.size(dim) - drop_length)
    stop_frame = min(seq.size(dim) - 1, start_frame + drop_length)
    slice1 = [slice(None)] * seq.dim()
    slice2 = [slice(None)] * seq.dim()
    slice1[dim] = slice(0, start_frame)
    slice2[dim] = slice(stop_frame, seq.size(dim))
    seq = torch.cat([seq[slice1], seq[slice2]], dim=dim)
    return seq


def random_speed_change(audio_tensor, step, sr, change_rate):
    frames = 0
    new_audios = []
    speed_factor = random.uniform(1 - change_rate, 1 + change_rate)
    while frames < audio_tensor.size(-1):
        # 生成随机速度因子，范围在0.8到1.2之间
        speed_factor = (speed_factor + random.uniform(1 - change_rate, 1 + change_rate)) / 2
        # 使用 torchaudio.sox_effects.apply_effects_tensor 来应用速度变换
        effects = [
            ["speed", f"{speed_factor}"],
            ["rate", f"{sr}"],
        ]
        # 应用效果并返回处理后的Tensor
        new_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_tensor[..., frames : frames + step], 16000, effects
        )
        new_audios.append(new_audio)
        frames += step
    if len(new_audios) > 0:
        return torch.cat((new_audios), -1)
    else:
        return audio_tensor
