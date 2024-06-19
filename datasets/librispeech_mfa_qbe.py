from math import ceil
from pathlib import Path
import random
from typing import Tuple, Union
from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio.functional as F
import torchaudio
from torchaudio.datasets.librispeech import FOLDER_IN_ARCHIVE, LIBRISPEECH, URL
import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import torchaudio
import torchaudio.functional as F
import random
from torch.nn.utils.rnn import pad_sequence
from utils import seq_util as util


def load_librispeech_audio(name, path, sr=16000, clip: tuple = None, max_frames=None):
    speaker_id, chapter_id, utterance_id = name.split("-")
    file_audio = os.path.join(path, speaker_id, chapter_id, name + ".flac")
    waveform, sample_rate = torchaudio.load(file_audio)
    if sample_rate != sr:
        waveform = F.resample(waveform, sample_rate, sr)
        sample_rate = sr
    if clip:
        start_frame = int(sr * clip[0])
        end_frame = int(sr * clip[1])
        if start_frame < 0:
            end_frame -= start_frame
            start_frame = 0
        elif end_frame > waveform.shape[-1]:
            start_frame -= end_frame - waveform.shape[-1]
            end_frame = waveform.shape[-1]
        waveform = waveform[:, start_frame:end_frame]
    if max_frames:
        waveform = torch.Tensor(librosa.util.fix_length(waveform, size=max_frames))
    return waveform, sample_rate


class UnlabeledLibriSpeechSegment(LIBRISPEECH):
    def __init__(
        self, root: str | Path, url: str, feat, segment_duration=(0.5, 2), query_duration=(0.4, 1)
    ) -> None:
        super().__init__(root, url)
        self.feat = feat if feat is not None else nn.Identity()
        self.sr = 16000
        self.dur = segment_duration
        self.q_dur = query_duration

    def __getitem__(self, n: int):
        waveform, sample_rate, _, _, _, _ = super().__getitem__(n)
        dur = min(
            int((random.random() * (self.dur[1] - self.dur[0]) + self.dur[0]) * self.sr),
            waveform.size(-1) - 1,
        )
        s_start = random.randrange(0, waveform.size(-1) - dur)
        s = waveform[:, s_start : s_start + dur]

        q_dur = min(
            int((random.random() * (self.q_dur[1] - self.q_dur[0]) + self.q_dur[0]) * self.sr),
            dur - 1,
        )
        q_start = random.randrange(0, dur - q_dur)
        q = s[:, q_start : q_start + q_dur]
        s = self.feat(s)
        q = self.feat(q)
        ext = torch.zeros(q.size(-1))
        ext[q_start : q_start + q_dur] = 1
        return s, q, ext

    @classmethod
    def collect_fn(cls, batch_data):
        list_s = []
        len_s = []
        for s in batch_data:
            # CFT->TFC
            list_s.append(s.transpose(0, -1))
            len_s.append(s.size(0))
        # BTFC->BCFT
        return pad_sequence(list_s, True).transpose(1, -1), torch.LongTensor(len_s)


class UnlabeledLibriSpeechSTD(LIBRISPEECH):
    def __init__(
        self, root: str | Path, url: str, feat, segment_duration=(0.5, 2), word_duration=(0.4, 1)
    ) -> None:
        super().__init__(root, url)
        self.feat = feat if feat is not None else nn.Identity()
        self.sr = 16000
        self.s_dur = segment_duration
        self.q_dur = word_duration

    def __getitem__(self, n: int):
        waveform, sample_rate, _, _, _, _ = super().__getitem__(n)
        if (waveform.size(-1) / self.sr) < self.s_dur[0]:
            waveform = util.zero_pad(waveform, int(self.sr * self.s_dur[0]))
        s_dur = min(
            int((random.random() * (self.s_dur[1] - self.s_dur[0]) + self.s_dur[0]) * self.sr),
            waveform.size(-1) - 1,
        )
        q_dur = min(
            int((random.random() * (self.q_dur[1] - self.q_dur[0]) + self.q_dur[0]) * self.sr),
            s_dur - 1,
        )
        s_start = random.randrange(0, waveform.size(-1) - s_dur)
        s = waveform[:, s_start : s_start + s_dur]
        q_start = random.randrange(0, s_dur - q_dur)
        q = s[:, q_start : q_start + q_dur]
        # 随机拉伸
        # src_idx=np.arange(q.size(-1))
        # random_length=int((random.random()*0.3+0.85)*q.size(-1))
        # if random_length>q.size(-1):
        #     dup_ids=np.random.choice(src_idx,random_length-q.size(-1),replace=True)
        #     tar_idx=np.append(src_idx,dup_ids)
        # else:
        #     tar_idx=np.random.choice(src_idx,random_length,replace=False)
        # tar_idx.sort()
        # q=q[...,tar_idx]
        feat_s = self.feat(s)
        # feat_q = self.feat(q)
        feat_q = q
        # ext = torch.zeros(feat_s.size(-1))
        # feat_ds_rate = feat_s.size(-1) / s.size(-1)
        # query_start_frame = int(q_start * feat_ds_rate)
        # query_end_frame = int((q_start + q_dur) * feat_ds_rate)
        # ext[query_start_frame:query_end_frame] = 1
        return feat_s, feat_q, s, q  # , ext

    @classmethod
    def collect_fn(cls, batch_data):
        list_s = []
        list_q = []
        # list_ext = []
        len_s = []
        len_q = []
        for s, q, _, _ in batch_data:
            # C,*,T -> T,*,C
            list_s.append(s.transpose(0, -1))
            list_q.append(q.transpose(0, -1))
            # list_ext.append(ext)
            len_s.append(s.size(0))
            len_q.append(q.size(0))
        return (
            pad_sequence(list_s, True).transpose(1, -1),
            pad_sequence(list_q, True).transpose(1, -1),
            torch.LongTensor(len_s),
            torch.LongTensor(len_q),
        )  # , pad_sequence(list_ext, True)


class MFA:
    def __init__(
        self, segment_duration=(0.5, 2), word_duration=(0.4, 1), window_shift=1, drop_last=False
    ) -> None:
        self.segment_duration = segment_duration
        self.word_duration = word_duration
        self.window_shift = window_shift
        self.drop_last = drop_last

    def top_words(self, top):
        if isinstance(top, int):
            top_words = [w[0] for w in self.words_count[:top]]
        elif isinstance(top, tuple):
            top_words = [w[0] for w in self.words_count[top[0] : top[1]]]
        else:
            raise Exception(f"top: {top} is not a valid param")
        return top_words

    def subset(self, words):
        mfa = MFA(self.segment_duration, self.word_duration, self.window_shift, self.drop_last)
        mfa.words_file = {}
        mfa.words_count = []
        mfa.aligns = {}
        if isinstance(words, float):
            words = int(len(self.words_count) * words)
        if isinstance(words, int):
            words = list(self.words_file.keys())[:words]
        if isinstance(words, list):
            for word, count in self.words_count:
                if word in words:
                    mfa.words_count.append((word, count))
                else:
                    mfa.words_count.append((word, 0))
            for word in words:
                if word in self.words_file.keys():
                    mfa.words_file[word] = self.words_file[word]
                else:
                    mfa.words_file[word] = []
            for file_name, seg in self.aligns.items():
                file_segs = []
                for seg_start, seg_end, contained_words in seg:
                    seg_words = []
                    for word_info in contained_words:
                        if word_info[0] in words:
                            seg_words.append(word_info)
                    if len(seg_words) > 0:
                        file_segs.append([seg_start, seg_end, seg_words])
                if len(file_segs) > 0:
                    mfa.aligns[file_name] = file_segs
        return mfa

    def parse_align(self, line, accept_words=None):
        """
        this method parse each line from the MFA align file, result will be save to the instance variable 'aligns', 'words_file' and 'words_count'\\
        aligns is a dictionary and has structure like:\\
        --------file_name\\
          |-----segment 1\\
              ...
          |-----segment k\\
            |---segment start time\\
            |---segment end time\\
            |---words\\
              |-(word,word_start,word_end)\\
                  ...
              |-(word,word_start,word_end)\\
        words_file is a dictionary and has structure like:\\
        -----word\\
          |--(file_name,start,end)\\
              ...
          |--(file_name,start,end)\\
        """
        # split line string and extract infomation, each line will be like "file_name" "words" "end_times"
        data = line.replace('"', "").split()
        wav_file_name = data[0]
        words = data[1].split(",")
        end_times = data[2].split(",")
        # get audio duration
        duration = max(float(end_times[-1]), self.segment_duration[0])
        # set start&end time and words list for each segments
        canditate_aligns = []
        last_time = 0
        while last_time < duration:
            segment_duration = self.segment_duration[0] + random.random() * (
                self.segment_duration[1] - self.segment_duration[0]
            )
            if self.drop_last and last_time + segment_duration > duration:
                break
            canditate_aligns.append([last_time, min(last_time + segment_duration, duration), []])
            last_time += segment_duration
        end_times.insert(0, "0")
        for word, start_time, end_time in zip(words, end_times[:-2], end_times[1:]):
            start_time, end_time = float(start_time), float(end_time)
            # drop words that are too long or too short
            if (
                end_time - start_time < self.word_duration[0]
                or end_time - start_time > self.word_duration[1]
            ):
                continue
            # ignore space
            if len(word) > 0 and (accept_words == None or word in accept_words):
                valid = False
                for i in range(len(canditate_aligns)):
                    if start_time > canditate_aligns[i][0]:
                        if end_time < canditate_aligns[i][1]:
                            word_seg_start = start_time - canditate_aligns[i][0]
                            word_seg_end = end_time - canditate_aligns[i][0]
                            canditate_aligns[i][2].append((word, word_seg_start, word_seg_end))
                            valid = True
                    else:
                        break
                if valid:
                    try:
                        self.words_file[word].append((data[0], start_time, end_time))
                    except:
                        self.words_file[word] = [(data[0], start_time, end_time)]
                    try:
                        self.words_count[word] += 1
                    except:
                        self.words_count[word] = 1
        for seg_start, seg_end, words in canditate_aligns:
            if len(words) > 0:
                self.aligns[wav_file_name] = self.aligns.get(wav_file_name, [])
                self.aligns[wav_file_name].append([seg_start, seg_end, words])

    def load_mfa(self, mfa_dir=None, accept_words=None):
        if mfa_dir is not None:
            self.mfa_path = mfa_dir
        else:
            mfa_dir = self.mfa_path
        # segment wavforms and list words in each part, can be index as (s_id,c_id,u_id,segments_id),
        # each data contains: [segment_start,segment_end, [words and time bounds in segemnts]]
        self.aligns = {}
        # save statics data of all words and sort them by times of shown
        self.words_count = {}
        # save files which contain a certain word
        self.words_file = {}
        for speaker in os.listdir(mfa_dir):
            speaker_dir = os.path.join(mfa_dir, speaker)
            for chapter in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter)
                align_path = os.path.join(
                    chapter_dir, "{}-{}.alignment.txt".format(speaker, chapter)
                )
                with open(align_path) as f:
                    datas = f.readlines()
                for line in datas:
                    self.parse_align(line, accept_words)
        # remove only one word
        for word in list(self.words_file.keys()):
            if self.words_count[word] <= 1:
                self.words_file.pop(word)
                self.words_count.pop(word)
        # remove empty aligns
        for key in list(self.aligns.keys()):
            valid_segs = []
            for seg in self.aligns[key]:
                valid_words = []
                for word in seg[-1]:
                    if word[0] in self.words_file.keys():
                        valid_words.append(word)
                seg[-1] = valid_words
                if len(valid_words) > 0:
                    valid_segs.append(seg)
            if len(valid_segs) > 0:
                self.aligns[key] = valid_segs
            else:
                self.aligns.pop(key)
        self.words_count = sorted(self.words_count.items(), key=lambda d: d[1], reverse=True)
        return self


class LibriSpeechData(LIBRISPEECH):
    def __init__(
        self,
        root,
        url="train-clean-100",
        segments=1,
        segment_duration=(0.5, 2),
        word_duration=(0.4, 2),
        window_shift=1,
        sr=16000,
        drop_last=True,
        match_rate=0.5,
        **kws,
    ):
        super().__init__(root, url=url)
        self.sr = sr
        self.segments = segments
        self.segment_duration = segment_duration
        self.match_rate = match_rate
        self.word_duration = word_duration
        self.mfa = MFA(segment_duration, word_duration, window_shift, drop_last).load_mfa(
            self._path
        )
        self.init_mfa(self.mfa)
        self.parse_args(**kws)

    def parse_args(self, **kws):
        for k, v in kws.items():
            print(f"not valid param:{k}={v}")

    def init_mfa(self, mfa: MFA):
        self.aligns = mfa.aligns
        self.words_file = mfa.words_file
        self.words = list(mfa.words_file.keys())
        self._walker = list(mfa.aligns.keys())

    def get_random_word_wav(self, word, context=False, exclude=False):
        """
        获取指定词的一段音频
        @context：若为False则返回边界清晰的音频，否则返回包含前后若干随机帧的音频
        @exclude：默认为False，若为True，则返回不包含word指定的词的任意音频
        """
        if exclude:
            select_word = random.choice(self.words)
            max_loop = 1000
            while select_word == word:
                if max_loop > 0:
                    select_word = random.choice(self.words)
                    max_loop -= 1
                else:
                    raise Exception(
                        f'Error: max tries reached while finding words exclude "{word}"!!!'
                    )
        query_file, query_start, query_end = random.choice(self.words_file[word])
        if context:
            query_length = self.segment_duration[0] + random.random() * (
                self.segment_duration[1] - self.segment_duration[0]
            )
            front_length = random.random() * (query_length - (query_end - query_start))
            query_start = max(0, query_start - front_length)
            query_end = query_start + query_length
        query_wav, sr = load_librispeech_audio(
            query_file, self._path, self.sr, (query_start, query_end)
        )
        return query_wav

    def __getitem__(self, n):
        audio_name = self._walker[n]
        wavform, sample_rate = load_librispeech_audio(audio_name, self._path, self.sr)
        results = []
        segments = random.choices(self.aligns[audio_name], k=self.segments)
        for seg_start, seg_end, contained_words in segments:
            seg_start = int(seg_start * sample_rate)
            seg_end = int(seg_end * sample_rate)
            seg_wav = wavform[:, seg_start:seg_end]
            results.append(self.get_segment_result(seg_wav, contained_words))
        return results

    def get_segment_result(self, seg_wav, contain_words):
        raise Exception("not implement method!")


class SequentialSTD(LibriSpeechData):
    """
    额外构造参数：
    @s_feat
    @s_len
    @q_feat
    @q_len
    @require_query_wav
    """

    def parse_args(self, s_feat=None, q_feat=None, require_query_wav=True, **kws):
        self.require_query_wav = require_query_wav
        self.search_feat = s_feat if s_feat else nn.Identity()
        self.query_feat = q_feat if q_feat else self.search_feat
        super().parse_args(**kws)

    def get_segment_result(self, seg_wav, contain_words):
        if random.random() < self.match_rate:
            query_word, query_start, query_end = random.choice(contain_words)
            ext = 1
        else:
            query_word = random.choice(self.words)
            query_start = 0
            query_end = 0
            ext = 0
            for w, s, e in contain_words:
                if w == query_word:
                    query_start = s
                    query_end = e
                    ext = 1
                    break
        if self.require_query_wav:
            query = self.get_random_word_wav(query_word)
        else:
            query = query_word
        search_feat = self.search_feat(seg_wav)
        query_feat = self.query_feat(query)
        query_start = int(query_start * 16000 / seg_wav.size(-1) * search_feat.size(-1))
        query_end = int(query_end * 16000 / seg_wav.size(-1) * search_feat.size(-1))
        # ext = torch.zeros(search_feat.size(-1))
        # ext[query_start:query_end] = 1
        return (search_feat, query_feat, query_word, ext, query_start, query_end, seg_wav, query)

    @classmethod
    def collect_fn(cls, batch_data):
        list_search = []
        list_search_length = []
        list_query = []
        list_query_length = []
        list_ext = []
        list_ts = []
        list_te = []
        for mini_batch in batch_data:
            for search, query, query_word, ext, query_start, query_end, _, _ in mini_batch:
                # C,*,T -> T,*,C
                list_search.append(search.transpose(0, -1))
                list_search_length.append(search.size(-1))
                list_query.append(query.transpose(0, -1))
                list_query_length.append(query.size(-1))
                list_ext.append(ext)
                list_ts.append(query_start)
                list_te.append(query_end)
        #  B, T,*,C -> B, C,*,T
        list_search = pad_sequence(list_search, batch_first=True).transpose(-1, 1)
        list_query = pad_sequence(list_query, batch_first=True).transpose(-1, 1)
        list_ext = torch.Tensor(list_ext)
        list_search_length = torch.LongTensor(list_search_length)
        list_query_length = torch.LongTensor(list_query_length)
        list_ts = torch.LongTensor(list_ts)
        list_te = torch.LongTensor(list_te)
        return (list_search, list_query, list_search_length, list_query_length), (
            list_ext,
            list_ts,
            list_te,
        )


class SingleWordSTD(SequentialSTD):
    def get_segment_result(self, seg_wav, contain_words):
        search_word, search_start, search_end = random.choice(contain_words)
        start_frame = int(search_start * self.sr)
        end_frame = int(search_end * self.sr)
        sub_wav = seg_wav[:, start_frame:end_frame]
        if random.random() < self.match_rate:
            query_word = search_word
            ext = 1
        else:
            all_words = list(self.words_file.keys())
            all_words.remove(search_word)
            query_word = random.choice(all_words)
            ext = 0
        if self.require_query_wav:
            query = self.get_random_word_wav(query_word)
        else:
            query = query_word
        search = util.zero_pad(self.search_feat(sub_wav), self.s_len)
        query = util.zero_pad(self.query_feat(query), self.q_len)
        search_id = self.words.index(search_word)
        query_id = self.words.index(query_word)
        return search, query, query_word, ext, search_id, query_id


class TripletSTD(LibriSpeechData):
    def parse_args(self, feat=nn.Identity(), f_len=-1, **kws):
        self.feat = feat
        self.f_len = f_len
        super().parse_args(kws)

    def get_segment_result(self, seg_wav, contain_words):
        keyword, _, _ = random.choice(contain_words)
        # achor
        achor = self.feat(seg_wav)
        # positive
        positive = self.get_random_word_wav(keyword, True)
        positive = self.feat(positive)
        # negtive
        negative = self.feat(self.get_random_word_wav(keyword, True, True))
        achor = util.zero_pad(achor, self.f_len)
        positive = util.zero_pad(positive, self.f_len)
        negative = util.zero_pad(negative, self.f_len)
        return achor, positive, negative

    @classmethod
    def collect_fn(cls, batch_data):
        list_a = []
        list_p = []
        list_n = []
        for mini_batch in batch_data:
            for achor, positive, negative in mini_batch:
                list_a.append(achor)
                list_p.append(positive)
                list_n.append(negative)
        return (torch.stack(list_a), torch.stack(list_p), torch.stack(list_n)), None

speakers=[103, 1034, 1040, 1069, 1081, 1088, 1098, 1116, 118, 1183, 1235, 1246, 125, 1263, 1334, 1355, 1363, 1447, 1455, 150, 1502, 1553, 1578, 1594, 1624, 163, 1723, 1737, 1743, 1841, 1867, 1898, 19, 1926, 196, 1963, 1970, 198, 1992, 200, 2002, 2007, 201, 2092, 211, 2136, 2159, 2182, 2196, 226, 2289, 229, 233, 2384, 2391, 2416, 2436, 248, 250, 2514, 2518, 254, 26, 2691, 27, 2764, 2817, 2836, 2843, 289, 2893, 2910, 2911, 2952, 298, 2989, 302, 307, 311, 3112, 3168, 32, 3214, 322, 3235, 3240, 3242, 3259, 328, 332, 3374, 3436, 3440, 3486, 3526, 3607, 3664, 3699, 3723, 374, 3807, 3830, 3857, 3879, 39, 3947, 3982, 3983, 40, 4014, 4018, 403, 405, 4051, 4088, 412, 4137, 4160, 4195, 4214, 426, 4267, 4297, 4340, 4362, 4397, 4406, 441, 4441, 445, 446, 4481, 458, 460, 4640, 4680, 4788, 481, 4813, 4830, 4853, 4859, 4898, 5022, 5049, 5104, 5163, 5192, 5322, 5339, 5390, 5393, 5456, 5463, 5514, 5561, 5652, 5678, 5688, 5703, 5750, 5778, 5789, 5808, 5867, 587, 60, 6000, 6019, 6064, 6078, 6081, 6147, 6181, 6209, 625, 6272, 6367, 6385, 6415, 6437, 6454, 6476, 6529, 6531, 6563, 669, 6818, 6836, 6848, 6880, 6925, 696, 7059, 7067, 7078, 7113, 7148, 7178, 7190, 7226, 7264, 7278, 730, 7302, 7312, 7367, 7402, 7447, 7505, 7511, 7517, 7635, 7780, 7794, 78, 7800, 7859, 8014, 8051, 8063, 8088, 8095, 8098, 8108, 8123, 8226, 8238, 83, 831, 8312, 8324, 839, 8419, 8425, 8465, 8468, 8580, 8609, 8629, 8630, 87, 8747, 8770, 8797, 8838, 887, 89, 8975, 909, 911]
class LibSpeaker(Dataset):
    def __init__(self,root,url,sr,feat=None,segment_len=2) -> None:
        super().__init__()
        audio_path=os.path.join(root,"LibriSpeech",url)
        self.walker=[p for p in Path(audio_path).rglob("*.flac")]
        self.sr=sr
        self.feat=feat if feat is not None else nn.Identity()
        self.seg_dur=segment_len
    def __len__(self):
        return len(self.walker)
    def __getitem__(self, index) :
        audio_path = self.walker[index]
        speaker_id = int(audio_path.parts[-1].split("-")[0])
        speaker_id=speakers.index(speaker_id)
        waveform, sr = load_librispeech_audio(audio_path, self._path, self.sr)
        if sr != self.sr:
            waveform = F.resample(waveform, sr, self.sr)
        if self.seg_dur is not None:
            seg_len=(self.seg_dur*self.sr)
            seg_num=ceil(waveform.size(-1)/seg_len)
            waveform=util.up_pad(waveform,seg_len,-1)
            waveform=waveform.view(*wav_feat.shape[:-1],seg_num,seg_len).transpose(0,-2)
        wav_feat=self.feat(waveform)
        return wav_feat
class SingleWordKWS(LibriSpeechData):
    def __init__(
        self,
        root,
        url="train-clean-100",
        seg_duration=(999, 999),
        word_duration=(0, 999),
        sr=16000,
        feat=nn.Identity(),
        top_words=None,
        **kws,
    ):
        super().__init__(root, url, 1, seg_duration, word_duration, sr, **kws)
        if top_words is not None:
            wlist=self.mfa.top_words(top_words)
            new_mfa=self.mfa.subset(wlist)
            self.init_mfa(new_mfa)
        self.feat = feat

    def init_mfa(self, mfa):
        self._walker = []
        self.words = list(mfa.words_file.keys())
        for word, files in mfa.words_file.items():
            word_id = self.words.index(word)
            for query_file, query_start, query_end in files:
                self._walker.append(
                    (
                        word,
                        word_id,
                        query_file,
                        int(query_start * self.sr),
                        int(query_end * self.sr),
                    )
                )

    def __getitem__(self, n: int):
        word, word_id, file_name, word_start, word_end = self._walker[n]
        speaker_id = int(file_name.split("-")[0])
        speaker_id=speakers.index(speaker_id)
        audio, sr = load_librispeech_audio(file_name, self._path, self.sr)
        audio = audio[:, word_start:word_end]
        word_feature = self.feat(audio)
        return word_feature, word_id, speaker_id

    @classmethod
    def collect_fn(cls, batch_data):
        features = []
        word_ids = []
        spk_ids = []
        for word_feature, word_id, speaker_id in batch_data:
            features.append(word_feature)
            word_ids.append(word_id)
            spk_ids.append(speaker_id)
        # list_feature = pad_sequence(features, True)
        list_feature = util.pad_seq(features)
        list_wid = torch.LongTensor(word_ids)
        list_sid = torch.LongTensor(spk_ids)
        return list_feature, list_wid, list_sid


if __name__ == "__main__":
    x1 = torch.randn((100, 40))
    x2 = torch.randn((105, 40))
    print(torch.nn.utils.rnn.pad_sequence([x1, x2], batch_first=True).shape)
