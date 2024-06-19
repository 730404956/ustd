import os
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import random
import torch
import librosa
from utils.audio import random_seg, load_audio


def sub_data_collect(batch):
    feats = []
    lengths = []
    names = []
    for feat, length, name in batch:
        feats.append(feat)
        lengths.append(length)
        names.append(name)
    return pad_sequence(feats), torch.IntTensor(lengths), names


class MedicalEval(Dataset):
    def __init__(
        self,
        audio_path,
        query_path,
        gt_file_path,
        s_feat=None,
        q_feat=None,
        balance_sample=True,
        seg=True,
        neg=1,
        search_length=(0.5, 2),
        query_length=(0.4, 2),
        sr=8000,
    ) -> None:
        super().__init__()
        self.balance_sample = balance_sample
        self.seg = seg
        self.sr = sr
        self.s_len = search_length
        self.s_feat = s_feat if s_feat else nn.Identity()
        self.q_feat = q_feat if q_feat else nn.Identity()
        self.audio_path = audio_path
        self.query_path = query_path

        self.init_data(gt_file_path, neg, query_length)

    def init_data(self, gt_file_path, neg, query_length=(0.4, 2)):
        self.data_maps = load_ground_truth(gt_file_path, self.query_path, query_length)
        self.positive_pairs = []
        self.all_audio_files = os.listdir(self.audio_path)
        self.all_query_files = os.listdir(self.query_path)

        self.data_length = len(self.all_audio_files) * len(self.all_query_files)
        self.length = int(len(self.data_maps) * (1 + neg))
        for k in self.data_maps.keys():
            search_name, query_name = k.split("#")
            i = self.all_audio_files.index(search_name + ".wav")
            j = self.all_query_files.index(query_name + ".wav")
            self.positive_pairs.append(i * len(self.all_query_files) + j)
        if neg == 0:
            self.data_sep = 1
        else:
            self.data_sep = self.data_length / (self.length - len(self.positive_pairs))

    def __getitem__(self, index):
        if self.balance_sample:
            if index < len(self.positive_pairs):
                index = self.positive_pairs[index]
            else:
                sep = int(self.data_sep)
                index = min(
                    self.data_length,
                    int((index - len(self.positive_pairs)) * self.data_sep)
                    + random.randint(0, sep),
                )
                if index in self.positive_pairs:
                    index = min(self.data_length, index + random.randint(-sep, sep))
        search_id = int(index / len(self.all_query_files))
        query_id = index % len(self.all_query_files)
        search_name = os.path.splitext(self.all_audio_files[search_id])[0]
        query_name = os.path.splitext(self.all_query_files[query_id])[0]
        key = f"{search_name}#{query_name}"
        tbeg, dur, _ = self.data_maps.get(key, (0, 0, 0))
        ext = 1 if dur > 0 else 0
        speech, len_s = load_audio(os.path.join(self.audio_path, search_name + ".wav"), self.sr)
        query, len_q = load_audio(os.path.join(self.query_path, query_name + ".wav"), self.sr)
        # 绝对位置表示为相对位置，即在待测语音中的开始位置占总长度的比例
        if self.seg:
            max_len = int(self.s_len[1] * self.sr)
            min_len = int(self.s_len[0] * self.sr)
            if dur == 0:
                keep = None
            else:
                keep = (int((tbeg) * self.sr), int((tbeg + dur) * self.sr))
            speech, (tbeg_r, tend_r) = random_seg(speech, (min_len, max_len), dim=-1, keep=keep)
        speech_feat = self.s_feat(speech)
        query_feat = self.q_feat(query)
        downsample_rate = speech.size(-1) / speech_feat.size(-1)
        tbeg_r = int(tbeg_r / downsample_rate)
        tend_r = min(speech_feat.size(-1), int(tend_r / downsample_rate))
        len_s = speech_feat.size(-1)
        len_q = query_feat.size(-1)
        return speech_feat, query_feat, len_s, len_q, ext, tbeg_r, tend_r, search_name, query_name

    def collect_fn(self, samples):
        speech, query, len_s, len_q, ext, t_start, duration, search_name, query_name = zip(*samples)
        return (
            (
                pad_sequence([s.transpose(0, -1) for s in speech], True).transpose(1, -1),
                pad_sequence([s.transpose(0, -1) for s in query], True).transpose(1, -1),
                torch.IntTensor(len_s),
                torch.IntTensor(len_q),
            ),
            (torch.Tensor(ext), torch.Tensor(t_start), torch.Tensor(duration)),
            search_name,
            query_name,
        )

    def __len__(self):
        return self.length

    def get_sampler(self):
        if self.balance_sample:
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )
        else:
            return None

    @property
    def sample_weights(self):
        """Sample weights to balance positive and negative data."""
        pos_num = len(self.positive_pairs)
        neg_num = self.length - pos_num
        return [1 / pos_num] * pos_num + [1 / neg_num] * neg_num


def get_sws2013(
    root,
    url,
    s_feat=None,
    q_feat=None,
    balance_sample=True,
    seg=True,
    neg=1,
    search_len=(0.5, 4),
    query_len=(0.4, 999),
    sr=8000,
):
    url_map = {
        "dev": ("dev_queries", "sws2013_dev/sws2013_dev.rttm"),
        "test": ("eval_queries", "sws2013_eval/sws2013_eval.rttm"),
    }
    audio_path = os.path.join(root, "sws2013Database_dev_eval", "Audio")
    query_path, gt_file_path = url_map[url]
    query_path = os.path.join(root, "sws2013Database_dev_eval", query_path)
    gt_file_path = os.path.join(root, "sws2013Database_dev_eval/scoring_atwv_sws2013", gt_file_path)
    return MedicalEval(
        audio_path,
        query_path,
        gt_file_path,
        s_feat,
        q_feat,
        balance_sample,
        seg,
        neg,
        search_len,
        query_len,
        sr,
    )


def get_quesst14(
    root,
    url,
    s_feat=None,
    q_feat=None,
    balance_sample=True,
    seg=True,
    neg=1,
    search_len=(0.5, 4),
    query_len=(0.4, 999),
    sr=8000,
):
    url_map = {
        "dev": ("dev_queries", "scoring/groundtruth_quesst14_dev/quesst14_dev.rttm"),
        "test": ("eval_queries", "scoring/groundtruth_quesst14_eval/quesst14_eval.rttm"),
    }
    audio_path = os.path.join(root, "quesst14Database", "Audio")
    query_path, gt_file_path = url_map[url]
    query_path = os.path.join(root, "quesst14Database", query_path)
    gt_file_path = os.path.join(root, "quesst14Database", gt_file_path)
    return MedicalEval(
        audio_path,
        query_path,
        gt_file_path,
        s_feat,
        q_feat,
        balance_sample,
        seg,
        neg,
        search_len,
        query_len,
        sr,
    )


def load_ground_truth(
    file_name,
    query_path,
    q_len_limits,
):
    datas = {}
    with open(file_name) as f:
        for l in f.readlines():
            content = l.split()
            if content[0] == "LEXEME" and content[5] != "NO_KEYWORD":
                search = content[1]
                query = content[5]
                time_start = float(content[3])
                duration = float(content[4])
                if duration < q_len_limits[0] or duration > q_len_limits[1]:
                    continue
                for i in range(1, 11):
                    new_query = query if i < 2 else query + f"_{i:0>2d}"
                    query_file_path = os.path.join(query_path, new_query + ".wav")
                    if not os.path.exists(query_file_path):
                        break  # 没有更多query，提前结束
                    q_len = librosa.get_duration(filename=query_file_path)
                    if q_len >= q_len_limits[0] and q_len <= q_len_limits[1]:
                        key = f"{search}#{new_query}"
                        datas[key] = (time_start, duration, q_len)
    return datas
