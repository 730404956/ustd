from torch.utils.data import Dataset
import torchaudio
import os
from torch.nn.utils.rnn import pad_sequence
from utils.feature.audio import Mel_Spec, MFCC
import torch
import random
from utils.seq_util import up_pad
import random


def read_phones(phones_lables_path):
    data = {}
    all_speakers = []
    with open(phones_lables_path, "r") as f:
        for line in f.readlines():
            splited_lines = line.strip().split(" ")
            audio_name = splited_lines[0]
            phones = [int(p) for p in splited_lines[1:]]
            data[audio_name] = phones
            speaker = audio_name.split("-")[0]
            if speaker not in all_speakers:
                all_speakers.append(speaker)
    return data, all_speakers


def split_dataset(
    lib_root, phones_lables_path, train_audios, test_audios, feat_type=None, samples=None
):
    if feat_type == "mfcc":
        feat = MFCC(win_length=320, hop_length=160)
    elif feat_type == "mel":
        feat = Mel_Spec(True,win_length=320, hop_length=160)
    else:
        feat = None
    data, all_speakers = read_phones(phones_lables_path)
    train_datas = {}
    dev_datas = {}
    test_datas = {}
    with open(train_audios) as f:
        for line in f.readlines():
            audio_name = line.strip()
            if random.random() < 0.9:
                train_datas[audio_name] = data[audio_name]
            else:
                dev_datas[audio_name] = data[audio_name]
    with open(test_audios) as f:
        for line in f.readlines():
            audio_name = line.strip()
            test_datas[audio_name] = data[audio_name]
    train_dataset = LibriPhoneSpeaker(lib_root, train_datas, all_speakers, feat, samples)
    dev_dataset = LibriPhoneSpeaker(lib_root, dev_datas, all_speakers, feat, samples)
    test_dataset = LibriPhoneSpeaker(lib_root, test_datas, all_speakers, feat, samples)
    return train_dataset, dev_dataset, test_dataset


def speakerid2index(speaker_id, speaker_ids: list):
    return speaker_ids.index(speaker_id)


class LibriPhoneSpeaker(Dataset):
    def __init__(self, audio_root, phones, all_speakers, feat=None, samples=None) -> None:
        super().__init__()
        self.all_speakers = all_speakers
        self.audio_root = audio_root
        self.data = phones
        self.audio_names = list(phones.keys())
        self.feat = feat
        self.samples = samples

    def __getitem__(self, index):
        audio_name = self.audio_names[index]
        phones = self.data[audio_name]
        speaker_id, chapter_id, audio_id = audio_name.split("-")
        audio_path = os.path.join(
            self.audio_root,
            "LibriSpeech",
            "train-clean-100",
            speaker_id,
            chapter_id,
            f"{audio_name}.flac",
        )
        audio_data, _ = torchaudio.load(audio_path)
        if self.feat is not None:
            audio_data = self.feat(audio_data)
            diff = audio_data.size(-1) % len(phones)
            audio_data = audio_data[..., :-diff]
        else:
            diff = int((audio_data.size(-1) / 160) - len(phones))
            right = audio_data.size(-1) - audio_data.size(-1) % 160 - 160 * diff
            audio_data = audio_data[..., :right]
        if self.samples is not None and len(phones) > int(self.samples / 160):
            phone_start = random.randrange(0, len(phones) - int(self.samples / 160))
            audio_start = None
            if self.feat is None:
                audio_start = phone_start * 160
                audio_data = audio_data[..., audio_start : audio_start + self.samples]
            else:
                audio_start = phone_start
                audio_data = audio_data[..., audio_start : audio_start + int(self.samples / 160)]
            phones = phones[phone_start : phone_start + int(self.samples / 160)]
        return audio_data, torch.Tensor(phones), speakerid2index(speaker_id, self.all_speakers)

    def __len__(self):
        return len(self.audio_names)


def collect_datas(batch_data):
    """return features(B,C,*,T), phones(B,T), speaker ids(B)

    Args:
        batch_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    audio_datas = []
    phone_list = []
    speaker_id_list = []
    for audio_data, phones, speaker_id in batch_data:
        audio_datas.append(audio_data.transpose(-1,0))
        phone_list.append(phones)
        speaker_id_list.append(speaker_id)
    audio_datas = pad_sequence(audio_datas,batch_first=True).transpose(-1,1)
    phone_list = pad_sequence(phone_list,batch_first=True)
    audio_datas = (audio_datas - audio_datas.mean()) / audio_datas.std()
    return audio_datas, phone_list, torch.Tensor(speaker_id_list)


if __name__ == "__main__":
    train_data, test_data = split_dataset(
        "/home/lixr/data",
        "/home/lixr/data/LibriSpeech_phones/converted_aligned_phones.txt",
        "/home/lixr/data/LibriSpeech_phones/train_split.txt",
        "/home/lixr/data/LibriSpeech_phones/test_split.txt",
    )
    datas = train_data.__getitem__(0)
    print(datas)
