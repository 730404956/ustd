
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import torch
import librosa
import numpy as np
import os
# from utils.seq_util import remove_duplicate


class AISHELL_base(Dataset):
    def __init__(self, root, subset="training", sr=16000) -> None:
        super().__init__()
        self.root = root
        self.sr = sr
        self.files = []
        self.trans = {}

        if subset in ["train", "training"]:
            subset = "train"
        elif subset in ["dev", "validating", "validation"]:
            subset = "dev"
        elif subset in ["test", "testing"]:
            subset = "test"
        else:
            raise Exception(f"no subset named {subset}")

        self.load_wav(os.path.join(root, "wav", subset))
        self.load_tag(os.path.join(root, "transcript"))

    def __read_wav__(self, path) -> np.ndarray:
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    def __read_tag_from_dir__(self, path) -> np.ndarray:
        with open(path, "r") as f:
            content = f.read()
        return content

    def __read_tag__(self, path) -> np.ndarray:
        f_name = os.path.split(path)[-1].split(".")[0]
        tag = self.trans[f_name]
        return tag

    def load_tag(self, root):
        for f in os.listdir(root):
            if f.endswith(".txt"):
                with open(os.path.join(root, f)) as txt:
                    for line in txt.readlines():
                        line = line.replace("\n", "")
                        spliter = line.index(" ")
                        label = line[spliter:]
                        label = zh_to_pinyin(label)
                        label = remove_duplicate(label)
                        label = label.split(" ")
                        label = spell2index(label)
                        self.trans[line[:spliter]] = label

    def load_wav(self, root):
        if os.path.isdir(root):
            for f in os.listdir(root):
                self.load_wav(os.path.join(root, f))
        else:
            name, post = os.path.splitext(root)
            if post == ".wav":
                self.files.append(root)

    def __getitem__(self, index) -> Tuple:
        wav = self.__read_wav__(self.files[index])
        tag = self.__read_tag__(self.files[index])
        return wav, tag

    def __len__(self):
        return len(self.files)


class AISHELL_lazy(AISHELL_base):
    def __init__(self, root, subset="training", sr=16000) -> None:
        super().__init__(root, subset, sr)

    def load_tag(self, root):
        for f in os.listdir(root):
            if f.endswith(".txt"):
                with open(os.path.join(root, f)) as txt:
                    for line in txt.readlines():
                        line = line.replace("\n", "")
                        spliter = line.index(" ")
                        label = line[spliter:]
                        self.trans[line[:spliter]] = label

    def __read_tag__(self, path) -> np.ndarray:
        f_name = os.path.split(path)[-1].split(".")[0]
        label = self.trans[f_name]
        label = zh_to_pinyin(label)
        label = remove_duplicate(label,output_type="str")
        label = label.split(" ")
        label = spell2index(label)
        return label


def get_data_loader(root, subset, batch_size=2, num_workers=0, feature_ext=None, load="normal"):
    if load == "lazy":
        data = AISHELL_lazy(root, subset)
    else:
        data = AISHELL_base(root, subset)

    def collect_datas(batch_data):
        # input: batch_data:[(data,label)*batch_size]
        # output: out:[tensor[*data.size,batch_size],tensor[*label.size,batch_size]]
        datas = []
        labels = []
        max_len = np.array([(x.shape[-1], len(y)) for x, y in batch_data]).max(axis=0)
        for data, label in batch_data:
            pad_width = ((0, max_len[0]-data.shape[-1]))
            pad_wav = np.pad(data, pad_width, constant_values=0)
            if feature_ext:
                pad_wav = feature_ext(pad_wav)
            datas.append(pad_wav)
            pad_width = ((0, max_len[1]-len(label)))
            labels.append(np.pad(label, pad_width, constant_values=0))
        return (torch.Tensor(np.array(datas)), torch.LongTensor(np.array(labels)))

    loader = DataLoader(data, collate_fn=collect_datas, shuffle=True,
                        batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return loader


if __name__ == "__main__":
    loader = get_data_loader("/home/lixr/data/data_aishell","train")
    for x, y in loader:
        print(x)
        print(y)
        exit()
