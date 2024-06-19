import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize(data: torch.Tensor, norm=1, dim=-2):
    if norm == None or norm == 0:
        return data
    max_v, _ = data.max(dim, keepdim=True)
    min_v, _ = data.min(dim, keepdim=True)
    return (data - min_v) / (max_v - min_v + 1e-8) * norm


def pad_seq(seq_list: list, dim=-1):
    if dim == 0:
        new_seq = pad_sequence(seq_list, True)
        return new_seq
    else:
        new_seq_list = [seq.transpose(0, dim) for seq in seq_list]
        new_seq = pad_sequence(new_seq_list, True)
        return new_seq.transpose(1, dim)


def pad_seq_mask(seq_list: list, dim=-1, max_len=None):
    if max_len is None:
        max_len = 0
        for seq in seq_list:
            if seq.size(dim) > max_len:
                max_len = seq.size(dim)
    new_seqs = []
    new_mask = []
    for seq in seq_list:
        seq, mask = zero_pad(seq, max_len, dim, need_mask=True)
        new_seqs.append(seq)
        new_mask.append(mask)

    return torch.stack(new_seqs), torch.stack(new_mask)


def zero_pad(seq: torch.Tensor, length, dim=-1, start=0, need_mask=False):
    """Pad sequence with 0 to fixed length

    Args:
        seq (torch.Tensor): _description_
        length (_type_): target length
        dim (int, optional): pad dim. Defaults to -1.
        start (float, optional): how many frames will be pad to the head of sequence(0~1),rest will be pad to the tail. Defaults to 0.5.
        need_mask (bool, optional): return pad mask or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    pad_shape = list(seq.shape)
    pad_shape[dim] = length - seq.shape[dim]
    if (dim >= 0 and dim < len(seq.shape)) or dim < -1:
        ending_shapes = pad_shape[dim + 1 :]
    else:
        ending_shapes = list()
    pad_head = int(pad_shape[dim] * start)
    pad_back = pad_shape[dim] - pad_head
    if type(seq) == np.ndarray:
        pad_mask = np.zeros(length)
        pad_mask[pad_head:pad_back] = 1
        if pad_shape[dim] > 0:
            pad_value_head = np.zeros((*pad_shape[:dim], pad_head, *ending_shapes))
            pad_value_back = np.zeros((*pad_shape[:dim], pad_back, *ending_shapes))
            paded_seq = np.concatenate((pad_value_head, seq, pad_value_back), axis=dim)
        else:
            seq = seq.swapaxix(dim, -1)
            seq = seq[..., int(abs(pad_shape[dim] / 2)) : int((pad_shape[dim] / 2).abs()) + length]
            paded_seq = seq.swapaxix(dim, -1)
    elif type(seq) == torch.Tensor:
        if pad_shape[dim] > 0:
            pad_mask = torch.zeros(length, device=seq.device)
            pad_mask[pad_head:-pad_back] = 1
            pad_value_head = torch.zeros((*pad_shape[:dim], pad_head, *ending_shapes), device=seq.device)
            pad_value_back = torch.zeros((*pad_shape[:dim], pad_back, *ending_shapes), device=seq.device)
            paded_seq = torch.cat((pad_value_head, seq, pad_value_back), dim=dim)
        else:
            pad_mask = torch.ones(length, device=seq.device)
            paded_seq = torch.narrow(seq, dim, int(abs(pad_shape[dim] / 2)), length)
    if need_mask:
        return paded_seq, pad_mask
    else:
        return paded_seq


def up_pad(seq: torch.Tensor, times: int, dim=-1):
    n = seq.size(dim) % times
    if n != 0:
        seq = zero_pad(seq, seq.size(dim) - n + times, dim=dim)
    return seq


def word2ebd(word, word_len):
    ebd = np.zeros((word_len, 26))
    for i in range(len(word)):
        try:
            ebd[i][ALPHABET.index(word[i])] = 1
        except:
            pass
    return ebd


def word2phoneme(word):
    pass


def remove_duplicate(list_a, output_type="list"):
    result = []
    last = None
    for x in list_a:
        if x != last:
            result.append(x)
            last = x
    if output_type == "list":
        return result
    elif output_type == "str":
        return "".join(result)
