from torch.utils.data import Dataset, WeightedRandomSampler
import torch
from torch.nn.utils.rnn import pad_sequence


class LibSWSData(Dataset):
    def __init__(self, lib_data, sws_data) -> None:
        super().__init__()
        self.lib_data = lib_data
        self.sws_data = sws_data

    def __len__(self):
        return len(self.lib_data)+len(self.sws_data)

    def __getitem__(self, n):
        len_lib = len(self.lib_data)
        if n < len_lib:
            return self.lib_data.__getitem__(n)
        else:
            return self.sws_data.__getitem__(n-len_lib)

    def get_sampler(self):
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )

    @property
    def sample_weights(self):
        p = len(self.sws_data.positive_pairs)
        n = self.sws_data.length-p
        l = len(self.lib_data)
        return [0.5/l]*l+[0.25/p]*p+[0.25/n]*n

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
            if isinstance(mini_batch[0], (list, tuple)):
                for search, query, query_word, ext, query_start, query_end in mini_batch:
                    # C,F,T -> T,F
                    list_search.append(search)
                    list_search_length.append(search.size(0))
                    list_query.append(query)
                    list_query_length.append(query.size(0))
                    list_ext.append(ext)
                    list_ts.append(query_start)
                    list_te.append(query_end)
            else:
                speech, query, len_s, len_q, ext, query_start, duration, search_name, query_name = mini_batch
                list_search.append(speech)
                list_search_length.append(len_s)
                list_query.append(query)
                list_query_length.append(len_q)
                list_ext.append(ext)
                list_ts.append(query_start)
                list_te.append(query_start+duration)

        list_search = pad_sequence(list_search, batch_first=True)
        list_query = pad_sequence(list_query, batch_first=True)
        list_search_length = torch.LongTensor(list_search_length)
        list_query_length = torch.LongTensor(list_query_length)
        list_ext = torch.LongTensor(list_ext)
        list_ts = torch.Tensor(list_ts)
        list_te = torch.Tensor(list_te)
        return (list_search, list_query, list_search_length, list_query_length), (list_ext, list_ts, list_te)
