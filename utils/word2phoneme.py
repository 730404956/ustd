import torch.nn as nn
import torch


class Word2Phone(nn.Module):
    def __init__(self, lexicon_path, fixed_len=-1) -> None:
        super().__init__()
        with open(lexicon_path)as f:
            datas = f.readlines()
        phonemes = {}
        for line in datas:
            lexicon = line.split()
            word = lexicon[0]
            for phone in lexicon[1:]:
                try:
                    phonemes[phone] += 1
                except:
                    phonemes[phone] = 1
        all_phonemes = sorted(phonemes.keys())
        self.phone_num = len(all_phonemes)
        word_phonemes = {}
        for line in datas:
            lexicon = line.split()
            word = lexicon[0]
            word_phonemes[word] = torch.zeros((len(lexicon)-1, self.phone_num))
            for idx, phone in enumerate(lexicon[1:]):
                word_phonemes[word][idx, all_phonemes.index(phone)] = 1
        self.word_phonemes = word_phonemes
        self.fixed_len = fixed_len

    def forward(self, word: str):
        word = word.upper()
        try:
            phoneme = self.word_phonemes[word]
        except:
            phoneme = torch.zeros((3, self.phone_num))
        if self.fixed_len > 0:
            pad_value = torch.zeros((self.fixed_len-phoneme.shape[0], self.phone_num))
            phoneme = torch.cat((phoneme, pad_value), dim=0)
        return phoneme
