import os
from torch.utils.data import Dataset, WeightedRandomSampler
import torchaudio
from torchaudio.functional import resample
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import random
import torch
import librosa
import json


def sub_data_collect(batch):
    feats = []
    lengths = []
    names = []
    for feat, length, name in batch:
        feats.append(feat)
        lengths.append(length)
        names.append(name)
    return pad_sequence(feats), torch.IntTensor(lengths), names


class SWS2013_Search(Dataset):
    def __init__(self, root, feat=None) -> None:
        super().__init__()
        self.feat = feat if feat else nn.Identity()
        self.audio_path = os.path.join(root, "quesst14Database", "Audio")
        self.files = os.listdir(self.audio_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = os.path.splitext(self.files[index])[0]
        wav,len = load_audio(self.audio_path, name)
        feat = self.feat(wav)
        return feat, feat.size(0), name


class SWS2013_Query(Dataset):
    def __init__(self, root, url, feat=None) -> None:
        super().__init__()
        self.feat = feat if feat else nn.Identity()
        if "dev" in url:
            self.query_path = os.path.join(root, "quesst14Database", "dev_queries")
        elif "test" in url:
            self.query_path = os.path.join(root, "quesst14Database", "eval_queries")
        else:
            raise Exception(f"url {url} not valid!!!")
        self.files = os.listdir(self.query_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = os.path.splitext(self.files[index])[0]
        wav,len = load_audio(self.query_path, name)
        feat = self.feat(wav)
        return feat, feat.size(0), name


class QUESST14(Dataset):
    def __init__(
        self,
        root,
        url,
        s_feat=None,
        q_feat=None,
        balance_sample=True,
        seg=True,
        neg=1,
        training_length=(0.4, 999),
        sr=8000,
        postfix=""
    ) -> None:
        super().__init__()
        self.url = url
        self.balance_smaple = balance_sample
        self.seg = seg
        self.sr = sr
        self.from_data_tag = False
        self.training_length = training_length
        self.s_feat = s_feat if s_feat else nn.Identity()
        self.q_feat = q_feat if q_feat else nn.Identity()
        self.audio_path = os.path.join(root, "quesst14Database", "Audio")
        if "dev" in url:
            self.query_path = os.path.join(root, "quesst14Database", "dev_queries"+postfix)
            gt_file_path = os.path.join(root, "quesst14Database/scoring/groundtruth_quesst14_dev", "quesst14_dev.rttm")
        elif "test" in url:
            self.query_path = os.path.join(root, "quesst14Database", "eval_queries"+postfix)
            gt_file_path = os.path.join(root, "quesst14Database/scoring/groundtruth_quesst14_eval", "quesst14_eval.rttm")
        else:
            raise Exception(f"url {url} not valid!!!")
        self.data_maps = load_ground_truth(
            gt_file_path, self.audio_path, self.query_path, training_length,True
        )
        self.positive_pairs = []
        self.all_audio_files = []
        self.all_query_files = []
        for f in os.listdir(self.audio_path):
            dur = librosa.get_duration(filename=os.path.join(self.audio_path, f), sr=8000)
            self.all_audio_files.append((f, dur))
        for f in os.listdir(self.query_path):
            dur = librosa.get_duration(filename=os.path.join(self.query_path, f), sr=8000)
            self.all_query_files.append((f, dur))
        self.all_audio_files = [n for n, d in sorted(self.all_audio_files, key=lambda x: x[-1])]
        self.all_query_files = [n for n, d in sorted(self.all_query_files, key=lambda x: x[-1])]

        self.data_length = len(self.all_audio_files) * len(self.all_query_files)
        self.length = len(self.data_maps) * (1 + neg) if neg > 0 else self.data_length
        for k in self.data_maps.keys():
            search_name, query_name = k.split("#")
            i = self.all_audio_files.index(search_name + ".wav")
            j = self.all_query_files.index(query_name + ".wav")
            self.positive_pairs.append(i * len(self.all_query_files) + j)
        self.data_sep = self.data_length / (self.length - len(self.positive_pairs))

    def from_data(self, search, query):
        self.from_data_tag = True
        self.cached_search = search
        self.cached_query = query

    def __getitem__(self, index):
        if self.balance_smaple:
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
        tbeg, dur,_ = self.data_maps.get(key, (0, 0,0))
        speech,len_s = load_audio(self.audio_path, search_name, self.sr)
        query,len_q = load_audio(self.query_path, query_name, self.sr)
        ext = 1 if dur > 0 else 0
        tbeg_r = tbeg * self.sr / speech.size(-1)
        tend_r = min(1.0, (tbeg + dur) * self.sr / speech.size(-1))
        speech = self.s_feat(speech)
        query = self.q_feat(query)
        # 绝对位置表示为相对位置，即在待测语音中的开始位置占总长度的比例
        if self.seg:
            (speech, tbeg_r, tend_r) = random_seg(speech, 160, 20, dim=0, keep=(tbeg_r, tend_r))
        len_s = speech.size(-1)
        len_q = query.size(-1)
        if len_q>self.sr*2:
            st=random.randrange(0,len_q-self.sr*2)
            query=query[...,st:st+self.sr*2]
        return speech, query, len_s, len_q, ext, tbeg, dur, search_name, query_name

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
        if self.balance_smaple:
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


def load_audio(folder, name, tgt_sr=None):
    audio_path = os.path.join(folder, name + ".wav")
    speech, sr = torchaudio.load(audio_path)
    length = speech.size(-1) / sr
    if tgt_sr is not None and sr != tgt_sr:
        speech = resample(speech, sr, tgt_sr)
    return speech, length


def random_seg(seq: torch.Tensor, max_len, min_len=0, dim=-1, keep=None):
    seq_len = seq.size(dim)
    # 不对短于设定最大长度的序列切割
    if seq_len < max_len:
        return seq, keep[0], keep[1]
    # 计算至少需要保留的长度
    if keep == None or keep == (0, 0):
        random_start = random.random()
        keep = (random_start, random_start)
    keep_beg = seq_len * keep[0]
    keep_end = seq_len * keep[1]
    keep_len = keep_end - keep_beg
    # 计算需要填充的长度
    pad_length = min(max_len - keep_len, min_len + (max_len - min_len) * random.random())
    left_pad = pad_length * random.random()
    right_pad = pad_length - left_pad
    rest_left = keep_beg - left_pad
    rest_right = seq_len - keep_end - right_pad
    if rest_left < 0:  # 左边填充过多
        left_pad += rest_left
        right_pad -= rest_left
    elif rest_right < 0:  # 右边填充过多
        left_pad -= rest_right
        right_pad += rest_right
    left = max(int(keep_beg - left_pad), 0)
    right = min(int(keep_end + right_pad), seq_len - 1)
    new_len = right - left
    if new_len < min_len:
        print()
    if dim == -1:
        seg_seq = seq[..., left:right]
    elif dim == 0:
        seg_seq = seq[left:right, ...]
    return seg_seq, left_pad / new_len, (keep_end - left) / new_len


def load_ground_truth(
    file_name,
    audio_path="Audio",
    query_path="dev_queries",
    keyword_length_limit=(0.2, 2),
    force_reload=False
):
    save_record_name = f"{file_name}.rec"
    if os.path.exists(save_record_name) and not force_reload:
        print(f"load saved data from {save_record_name}")
        with open(save_record_name, "r") as f:
            datas = json.load(f)
    else:
        datas = {}
        with open(file_name) as f:
            for l in f.readlines():
                content = l.split()
                if content[0] == "LEXEME":
                    search = content[1]
                    query = content[5]
                    time_start = float(content[3])
                    duration = float(content[4])
                    if duration < keyword_length_limit[1] and duration > keyword_length_limit[0]:
                        if query != "NO_KEYWORD":
                            for i in range(1, 11):
                                new_query = query if i < 2 else query + f"_{i:0>2d}"
                                if not os.path.exists(os.path.join(query_path, new_query + ".wav")):
                                    break  # 没有更多query，提前结束
                                _,wav_len = load_audio(query_path, new_query)
                                key = f"{search}#{new_query}"
                                datas[key] = (time_start, duration, wav_len)
        # dumps 将数据转换成字符串
        info_json = json.dumps(datas, sort_keys=False, indent=4, separators=(",", ": "))
        with open(save_record_name, "w") as f:
            f.write(info_json)
    new_datas = {}
    for k, v in datas.items():
        wav_len = v[-1]
        # 防止过长或过短的片段被读取
        if wav_len >= keyword_length_limit[0] and wav_len <= keyword_length_limit[1]:
            new_datas[k] = v#(v[0], v[1])
        else:
            pass
    return new_datas


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    datas = QUESST14("/home/lxr/data", "test", balance_sample=True, neg=1)
    datas.__getitem__(0)
    datas.__getitem__(30000)
    length = []
    for d in datas:
        l = d[0].size(-1)
        if l < 2000:
            print(d[-2], l)
        length.append(l)
    length.sort()
    print(length[:100])
    plt.hist(length)
    plt.savefig("h.png")
    # batch = []
    # for i in datas.positive_pairs:
    #     dt = datas.__getitem__(i)
    #     batch.append(dt)
    # res = datas.collect_fn(batch)
    # print(res)
