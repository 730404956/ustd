from utils.seq_util import zero_pad
import os
import numpy as np
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from audiomentations import AddBackgroundNoise
from torchaudio import transforms
from utils.augmentaion.audio import resample, spec_augment, time_shift
from torchaudio.datasets.utils import _load_waveform

from utils.audio import (
    load_audio,
    random_seg,
    remove_silent,
    list_audio,
    NoiseAdder,
    random_drop,
    random_speed_change,
)

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SAMPLE_RATE = 16000


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def _get_speechcommands_metadata(filepath: str, path: str):
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    return relpath, SAMPLE_RATE, label, speaker_id, utterance_number


class log_mel(torch.nn.Module):
    def __init__(self, f_size, sr) -> None:
        super().__init__()
        self.EPS = 1e-9
        self.mel = transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, f_max=8000, n_mels=f_size)

    def forward(self, x):
        y = self.mel(x)
        y = (y + self.EPS).log2()
        return y


def get_speakers(path):
    speaker_ids = {}
    for word in LABELS:
        audio_path = os.path.join(path, word)
        if os.path.exists(audio_path):
            for file in os.listdir(audio_path):
                speaker_name = file.split("_")[0]
                speaker_ids[speaker_name] = speaker_ids.get(speaker_name, 0) + 1
    speaker_ids = list(speaker_ids.keys())
    return speaker_ids


def one_hot(elem, seq: list):
    encodeing = np.zeros(len(seq))
    try:
        encodeing[seq.index(elem)] = 1
    except:
        pass
    return encodeing


def get_idx(elem, seq: list, default=-1):
    id = default
    try:
        id = seq.index(elem)
    except:
        pass
    return id


def split_data(data_list, ex_words, ex_speakers, rate=0.5):
    train_set = []
    exw_set = []
    exs_set = []
    for data in data_list:
        exs = exw = False
        for ew in ex_words:
            if ew in data:
                exw = True
                break
        for es in ex_speakers:
            if es in data:
                exs = True
                break
        if not (exs or exw):
            train_set.append(data)
        elif exs and not exw:
            exs_set.append(data)
        elif exw and not exs:
            exw_set.append(data)
    random.shuffle(exw_set)
    random.shuffle(exs_set)
    ss = int(len(exs_set) / 2)
    sw = int(len(exw_set) / 2)
    return train_set, exw_set[:sw], exw_set[sw:], exs_set[:ss], exs_set[ss:]


LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]


class SCData(Dataset):
    def get_metadata(self, n: int):
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to the audio
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)

    def __len__(self) -> int:
        return len(self._walker)

    def __init__(
        self,
        root,
        subset="training",
        augment=[],
        feat=torch.nn.Identity(),
        url="speech_commands_v0.02",
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        walker=None,
    ) -> None:
        if subset is not None and subset not in ["training", "validation", "testing", "all"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            pass

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)
        folder_in_archive = os.path.join(folder_in_archive, url)
        self._path = os.path.join(root, folder_in_archive)
        self.speaker_ids = get_speakers(self._path)
        if walker is not None:
            self._walker = walker
        elif not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path or set `download=True` to download it"
            )

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        elif subset == "all":
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]
        self.feature_ext = feat if feat is not None else torch.nn.Identity()
        self.augment = augment
        self.bg_adder = NoiseAdder(os.path.join(self._path, "_background_noise_"), 10)

    def __getitem__(self, n: int):
        metadata = self.get_metadata(n)
        # C,T
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        # waveform, sample_rate, label, speaker_id, utterance_number
        wav, sr, label, sid, uid = (waveform,) + metadata[1:]
        if "noise" in self.augment:
            wav = self.bg_adder(wav)
        if "speed" in self.augment:
            wav = random_speed_change(wav, int(sr * 0.35), sr, 0.2)
        if "resample" in self.augment:
            wav, sr = resample(wav, sr, 0.85, 1.15)
        wav = self.feature_ext(wav)
        if "spec_aug" in self.augment:
            wav = spec_augment(np.array(wav), 2, 25, 2, 7)
        y = get_idx(label, LABELS)
        return wav, y, self.speaker_ids.index(sid)


def collect_datas(batch_data):
    # input: batch_data:[(data,label)*batch_size]
    # output: out:[tensor[*data.size,batch_size],tensor[*label.size,batch_size]]
    datas = []
    sids = []
    labels = []
    max_length = np.array([x.size()[-1] for x, _, _ in batch_data]).max()
    for data, label, sid in batch_data:
        datas.append(zero_pad(data, max_length, -1).squeeze())
        labels.append(label)
        sids.append(sid)
    return torch.stack(datas), torch.LongTensor(np.array(labels)), torch.LongTensor(np.array(sids))
