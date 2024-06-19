import random
import torchaudio.datasets.librispeech as librispeech
import os
import torch
import numpy as np
from torchaudio.transforms import MFCC
import numba
from torch.utils.data import DataLoader

from utils.seq_util import word2ebd, zero_pad

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@numba.jit(nopython=True, cache=True)
def get_ext(word, aligns, c):
    ext = np.zeros(c)
    t = np.zeros(c)
    dt = np.zeros(c)
    block_duration = aligns[-1][-1]/c
    if c > 1:
        for w, t1, t2 in aligns:
            if w == word:
                for i in range(c):
                    if t1 > block_duration*i and t2 < block_duration*(i+1):
                        ext[i] = 1
                        t[i] = (t1+t2)/2-block_duration*i
                        dt[i] = (t2-t1)
    else:
        for w, t1, t2 in aligns:
            if w == word:
                ext[0] = 1
                t[0] = (t1+t2)/2
                dt[0] = (t2-t1)
                break

    return ext, t, dt


def load_align(path):
    align = {}
    words = {}
    words_count = []
    with open(path)as f:
        for line in f.readlines():
            data = line.split()
            wav_file_name = data[0].split("-")[-1]
            align[wav_file_name] = []
            words[wav_file_name] = []
            trans = data[1].replace("\"", "").split(",")
            end_times = data[2].replace("\"", "").split(",")
            end_times.insert(0, "0")
            for word, start_time, end_time in zip(trans, end_times[:-2], end_times[1:]):
                if len(word) > 0:
                    align[wav_file_name].append((word, float(start_time), float(end_time)))
                    words[wav_file_name].append(word)
                    words_count.append(word)
    return align, words, words_count


class LibriSpeechMFA(librispeech.LIBRISPEECH):
    def __init__(self, root, c, url="train-clean-100", feature="mfcc", feature_size=40, feature_len=800, query_len=20):
        super().__init__(root, url=url)
        self.c = c
        self.f_size = feature_size
        self.f_len = feature_len
        self.q_len = query_len
        if feature == "mfcc":
            self.feature_ext = MFCC(n_mfcc=feature_size)
        else:
            self.feature_ext = torch.nn.Identity()
        self.load_mfa(self._path)

    def load_mfa(self, mfa_dir):
        self.aligns = {}
        self.words = {"count": {}}
        for speaker in os.listdir(mfa_dir):
            self.aligns[speaker] = {}
            self.words[speaker] = {}
            speaker_dir = os.path.join(mfa_dir, speaker)
            for chapter in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter)
                align_path = os.path.join(
                    chapter_dir, '{}-{}.alignment.txt'.format(speaker, chapter))
                align, words, count = load_align(align_path)
                self.aligns[speaker][chapter] = align
                self.words[speaker][chapter] = words
                for w in count:
                    try:
                        self.words["count"][w] += 1
                    except:
                        self.words["count"][w] = 1
        self.words["count"] = sorted(self.words["count"].items(),  key=lambda d: d[1], reverse=True)

    def __getitem__(self, n):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = super().__getitem__(n)
        speaker_id, chapter_id, utterance_id = str(speaker_id), str(chapter_id), f"{utterance_id:0>4}"
        sub_wavs = torch.chunk(waveform, self.c, dim=-1)
        features = [self.feature_ext(wav) for wav in sub_wavs]
        paded_feat = [np.pad(f, ((0, 0), (0, 0), (0, self.f_len-f.shape[-1])), 'constant', constant_values=0) for f in features]
        try:
            word_ext_times = self.aligns[speaker_id][chapter_id][utterance_id]
            words = self.words[speaker_id][chapter_id][utterance_id]
            word = random.choice(words)
            ext, t, dt = get_ext(word, word_ext_times, self.c)
        except:
            word = self.words["count"][random.randint(0, 30)][0]
            ext = np.zeros(self.c)
            t = np.zeros(self.c)
            dt = np.zeros(self.c)
        word_ebd = word2ebd(word, self.q_len)
        return (paded_feat, word_ebd), (ext, t, dt)

    def get_sample(self, n):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = super().__getitem__(n)
        speaker_id, chapter_id, utterance_id = str(speaker_id), str(chapter_id), f"{utterance_id:0>4}"
        sub_wavs = torch.chunk(waveform, self.c, dim=-1)
        # features = [self.feature_ext(wav) for wav in sub_wavs]
        # paded_feat = [np.pad(f, ((0, 0), (0, 0), (0, self.f_len-f.shape[-1])), 'constant', constant_values=0) for f in features]
        try:
            word_ext_times = self.aligns[speaker_id][chapter_id][utterance_id]
            words = self.words[speaker_id][chapter_id][utterance_id]
            word = words[random.randint(0, len(words)-1)]
            ext, t, dt = get_ext(word, word_ext_times, self.c)
        except:
            word = self.words["count"][random.randint(0, 30)][0]
            ext = np.zeros(self.c)
            t = np.zeros(self.c)
            dt = np.zeros(self.c)
        # word_ebd = word2ebd(word, self.q_len)
        t_start = ext*(t-dt/2)
        t_end = ext*(t+dt/2)
        return sub_wavs, word, t_start, t_end


def collect_datas(batch_data):
    # input: batch_data:[(data,label)*batch_size]
    # output: out:[tensor[*data.size,batch_size],tensor[*label.size,batch_size]]
    feats = []
    words = []
    exts = []
    ts = []
    dts = []
    for (paded_feat, word_ebd), (ext, t, dt) in batch_data:
        for f in paded_feat:
            feats.append(f)
        words.append(word_ebd)
        exts.append(ext)
        ts.append(t)
        dts.append(dt)
    feats = torch.Tensor(np.array(feats))
    words = torch.Tensor(np.array(words))
    exts = torch.Tensor(np.array(exts))
    ts = torch.Tensor(np.array(ts))
    dts = torch.Tensor(np.array(dts))
    return (feats, words), (exts, ts, dts)


if __name__ == "__main__":
    dataset = LibriSpeechMFA("/home/lixr/data", 3)
    loader = DataLoader(dataset, collate_fn=collect_datas, batch_size=16)
    # (features, word_ebd), (ext, t, dt) = dataset.__getitem__(1)
    # print(features[0].shape)
    # print(word_ebd.shape)
    # print(ext)
    for (paded_feat, word_ebd), (ext, t, dt) in loader:
        print(paded_feat.shape)
        print(word_ebd.shape)
        print(ext.shape)
        exit()
