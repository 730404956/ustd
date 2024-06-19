import argparse
import torch
import torch.nn as nn

from core.BaseSolver import BaseSolver
from model.STD_Model import STD_Model
from datasets.librispeech_mfa_qbe import SequentialSTD
from utils.criteria import accuracy, ap, recall
from utils.function_warper import warp
from torch.utils.data import DataLoader
import random
import copy
from utils.seq_util import up_pad
from datasets.sws2013 import get_sws2013,get_quesst14
from datasets.UnlabeledData import UnlabeledSTD, STDDataConstructor
from utils.feature.audio import Mel_Spec


def get_std_loaders(feat, batch_size, match=0.5):
    root = "/home/lxr/data"
    seg_dur = (3, 3)
    data = SequentialSTD(root, "dev-clean", 5, seg_dur, s_feat=feat, q_feat=feat, match_rate=match)
    dev_loader = DataLoader(data, batch_size, True, collate_fn=data.collect_fn, num_workers=4)
    data = SequentialSTD(root, "test-clean", 5, seg_dur, s_feat=feat, q_feat=feat, match_rate=match)
    test_loader = DataLoader(data, batch_size, True, collate_fn=data.collect_fn, num_workers=4)
    return dev_loader, test_loader


def get_sws_loaders(feat, batch_size):
    root = "/home/lxr/data"
    seg_dur = (3, 3)
    data = get_sws2013(root, "dev", s_feat=feat, q_feat=feat, search_len=seg_dur, sr=16000)
    dev_loader = DataLoader(
        data,
        batch_size,
        sampler=data.get_sampler(),
        collate_fn=data.collect_fn,
        num_workers=4,
        drop_last=True,
    )
    data = get_sws2013(root, "test", s_feat=feat, q_feat=feat, search_len=seg_dur, sr=16000)
    test_loader = DataLoader(
        data,
        batch_size,
        sampler=data.get_sampler(),
        collate_fn=data.collect_fn,
        num_workers=4,
        drop_last=True,
    )
    return dev_loader, test_loader

def get_quesst_loaders(feat, batch_size):
    root = "/home/lxr/data"
    seg_dur = (3, 3)
    data = get_quesst14(root, "dev", s_feat=feat, q_feat=feat, search_len=seg_dur, sr=16000)
    dev_loader = DataLoader(
        data,
        batch_size,
        sampler=data.get_sampler(),
        collate_fn=data.collect_fn,
        num_workers=4,
        drop_last=True,
    )
    data = get_quesst14(root, "test", s_feat=feat, q_feat=feat, search_len=seg_dur, sr=16000)
    test_loader = DataLoader(
        data,
        batch_size,
        sampler=data.get_sampler(),
        collate_fn=data.collect_fn,
        num_workers=4,
        drop_last=True,
    )
    return dev_loader, test_loader


def get_rec_loaders(root, feat, batch_size):
    seg_dur = (4, 4)
    noise_path = "/home/lxr/data/SpeechCommands/speech_commands_v0.02/_background_noise_"
    data = UnlabeledSTD(
        root, 16000, None, None, seg_dur,(0.4,3), noise_path=noise_path, add_noise_on=(False, False)
    )
    train_loader = DataLoader(data, batch_size, True, collate_fn=data.collect_fn, num_workers=16)
    return train_loader


class Solver(BaseSolver):
    def __init__(self, data_con, base_path, name, device="cpu", init_seed=None) -> None:
        super().__init__(base_path, name, device, init_seed)
        self.ds = 8
        self.data_src = "vc_cls"
        self.data_con = data_con

    def __patch_data__(self, model, data):
        if self.data_src == "std":
            s, q, len_s, len_q = data[0]
            e, si, qi = data[1]
            s = up_pad(s, self.ds).squeeze()
            q = up_pad(q, self.ds).squeeze()
            ext = self.to_device(e.long())
            s = self.to_device(s)
            q = self.to_device(q)
            return (s, q, len_s, len_q), ext
        elif self.data_src == "wav_std":
            s, q, len_s, len_q = data[0]
            e, si, qi = data[1]
            s = self.to_device(s)
            q = self.to_device(q)
            with torch.no_grad():
                s = self.freevc.get_c(s)
                q = self.freevc.get_c(q)
            s = up_pad(s, self.ds).squeeze()
            q = up_pad(q, self.ds).squeeze()
            ext = self.to_device(e.long())
            return (s, q, len_s, len_q), ext
        elif self.data_src == "aug_std":
            s, q, len_s, len_q = data[0]
            e, si, qi = data[1]
            with torch.no_grad():
                q = q.squeeze().to(0)
                qs2 = self.freevc.get_s(q)
                qc2 = self.freevc.get_c(q)
                syn_q = self.freevc.syn(qc2, qs2, None)
                q = self.feat(syn_q).squeeze()
            s = up_pad(s, self.ds).squeeze()
            q = up_pad(q, self.ds).squeeze()
            ext = self.to_device(e.long())
            s = self.to_device(s)
            q = self.to_device(q)
            return (s, q, len_s, len_q), ext
        elif self.data_src == "rec_std":
            s, q, len_s, len_q = data
            data_con = self.to_device(self.data_con)
            s = self.to_device(s)
            q = self.to_device(q)
            with torch.no_grad():
                s, q, len_s, len_q, ext = data_con(s, q, len_s, len_q)
            s = up_pad(s, self.ds).squeeze()
            q = up_pad(q, self.ds).squeeze()
            return (s, q, len_s, len_q), ext

    def __back_patch__(self, model_output, ground_truth):
        if "std" in self.data_src:
            pred = model_output["pred"]  # .reshape((-1, 2))
            return {
                # "gt": ground_truth[0],
                "ext": ground_truth,
                "pred": pred,
                "pe": model_output["ext"],
            }
        elif self.data_src == "vc_rec":
            return model_output
        elif self.data_src == "vc_cls":
            phones, speaker_id = ground_truth
            return {"pid": phones, "sid": speaker_id, **model_output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", type=int, default=40)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-3)
    parser.add_argument("-fd", "--feature_dim", type=int, default=40)
    parser.add_argument("-ls", "--latent_size", type=int, default=32)
    args = parser.parse_args()

    EPOCH = args.epoch
    F_DIM = args.feature_dim
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    DEVICE = 3
    LATENT_SIZE = args.latent_size
    feat = Mel_Spec(
        True, n_feat=F_DIM, win_length=int(16000 * 0.05), hop_length=int(16000 * 0.0125)
    )
    noise_path = "/home/lxr/data/SpeechCommands/speech_commands_v0.02/_background_noise_"
    config={"RR":True,"CP":True,"NS":True,"ST":False}
    data_con = STDDataConstructor(16000, copy.deepcopy(feat), noise_path,**config)
    solver = Solver(
        data_con,
        "/data1/lxr_results/",
        f"std_RR+NS+CP",
        device=DEVICE,
    )
    solver.reporter.set_value("args", args)
    sws_dev, sws_test = get_quesst_loaders(feat, BATCH_SIZE)
    lib_dev, lib_test = get_std_loaders(feat, BATCH_SIZE)
    unsup_data_path = [
        "/home/lxr/data/LibriSpeech/train-clean-100",
        # "/home/lxr/data/LibriSpeech/train-clean-360",
        # "/home/lxr/data/LibriSpeech/dev-clean",
        # "/home/lxr/data/LibriSpeech/test-clean",
        "/home/lxr/data/quesst14Database/Audio",
        "/home/lxr/data/sws2013Database_dev_eval/Audio",
        "/home/lxr/data/MSLRL/gu-in-Train",
        "/home/lxr/data/MSLRL/ta-in-Train",
        "/home/lxr/data/MSLRL/te-in-Train",
        # "/home/lxr/data/LibriSpeech/dev-clean",
    ]
    rec_train = get_rec_loaders(unsup_data_path, feat, BATCH_SIZE)
    # rec_train = get_rec_loaders("/home/lxr/data/LibriSpeech/train-clean-360/", feat, BATCH_SIZE, ".flac")
    score_std = [
        {"name": "acc", "func": warp(accuracy, ["pe", "ext"])},
        {"name": "ap", "func": warp(ap, ["pe", "ext"])},
        {"name": "recall", "func": warp(recall, ["pe", "ext"])},
    ]
    loss_std = [
        {"name": "l_acc", "func": warp(nn.CrossEntropyLoss(), ["pe", "ext"]), "weight": 1},
    ]

    def set_std_rec():
        solver.data_src = "rec_std"

    def set_std_aug():
        solver.data_src = "aug_std"

    def set_std():
        solver.data_src = "std"


    std_model = STD_Model(F_DIM, LATENT_SIZE)
    solver.add_model(
        "std_model",
        std_model,
        # init_path="/data1/lxr_results/results_202310/162026_std_with_WavLM_hybrid_data_aug/exp1708-std/train_std_best_model.torch"
    )
    solver.add_stage("train_std", "std_model", rec_train, score_std, loss_std, lr=LR)
    solver.add_eval_stage("sws_dev", "std_model", sws_dev, score_std)
    solver.add_eval_stage("sws_eval", "std_model", sws_test, score_std)
    solver.add_eval_stage("lib_dev", "std_model", lib_dev, score_std)
    solver.add_eval_stage("lib_eval", "std_model", lib_test, score_std)
    solver.add_callback_stage("set_std_rec", set_std_rec)
    solver.add_callback_stage("set_std_aug", set_std_aug)
    solver.add_callback_stage("set_std", set_std)
    solver.add_load_stage("load_best_vccls", "vc_model", "vc_dev")
    solver.add_load_stage("load_best_std", "std_model", "std_dev")
    for i in range(3):
        seed = random.randrange(0, 9999)
        stage_std = ["set_std", "train_std"] * 10 + ["set_std_aug", "std_dev", "std_eval"]
        stage_std2 = ["set_std2", "train_std", "std_dev"] * 40 + ["std_eval"]
        stage_rec_std = (
            ["set_std", "lib_dev", "sws_dev"] * 0
            + ["set_std_rec", "train_std", "set_std", "lib_dev", "sws_dev"] * 25
            + ["set_std", "lib_eval", "sws_eval"]
        )
        tag = f"exp{seed}-std"
        solver.start(stage_rec_std, tag, init_seed=seed)
    print("all finished")
