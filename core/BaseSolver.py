import datetime
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

from .model_init import load_model_state, ModelInitor
from .Caculator import Caculator, NanOrInfException
from .reporter import TAG_CONFIG, TAG_RUNNING, TAG_MOD, Reporter
from .ResultsHolder import ResultsHolder
import copy

BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
NAME_TRAIN = "train"
NAME_DEV = "dev"
NAME_EVAL = "test"


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(obj, device):
    if isinstance(device, list):
        target_device = device[0]
        if isinstance(obj, nn.Module) and not isinstance(obj, nn.DataParallel):
            obj = nn.DataParallel(obj, device)
    else:
        target_device = device
    obj = obj.to(target_device)
    return obj


class BaseSolver:
    best_model_name = "best_model.torch"
    recent_model_name = "most_recent_model.torch"
    max_retry = 10

    def __init__(self, base_path, name, device="cpu", init_seed=None) -> None:
        self.base_path = base_path
        self.set_save_path(name)
        if init_seed is None:
            init_seed = random.randrange(1, 9999)
        self.reporter = Reporter()
        self.data = ResultsHolder()
        self.set_seed(init_seed, "init")
        self.log_init = False
        self.stages = {}
        self.models = {}
        self.lazy_logging_queue = []
        # default
        self.file_tag = ""
        self.device = device
        self.stages = {}

    def set_seed(self, seed, name=""):
        seed_torch(seed)
        self.reporter.set_value(f"{name} seed", seed)

    def to_device(self, obj):
        return to_device(obj, self.device)

    def __save__(self, obj, path):
        """将对象保存到创建时声明的路径下

        Args:
            obj (any): 要保存的对象
            path (str): 保存的文件路径
        """
        if obj is None:
            return
        if isinstance(obj, np.ndarray):
            obj.tofile(path)
        elif isinstance(obj, list):
            np.array(obj).tofile(path)
        elif isinstance(obj, torch.nn.Module):
            if isinstance(obj, nn.DataParallel):
                torch.save(obj.module.state_dict(), path)
            else:
                torch.save(obj.state_dict(), path)
        else:
            np.save(path, obj)

    def msg(self, content, level=logging.INFO, print_console=False, lazy_mode=False):
        if print_console:
            print(content)
        if lazy_mode:
            self.lazy_logging_queue.append((content, level))
            return
        while len(self.lazy_logging_queue) > 0:
            content_, level_ = self.lazy_logging_queue.pop(0)
            self.msg(content_, level_)
        if not self.log_init:
            # logging setting
            self.logger = logging.getLogger("solver")
            log_handler = logging.FileHandler(self.get_save_path("solver.log", True), "w")
            log_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-6s : - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.setLevel(logging.DEBUG)
            log_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(log_handler)
            self.log_init = True
        self.logger.log(level, "{}|{}".format(self.file_tag, content))

    def set_save_path(self, name=None, use_time=True):
        if use_time:
            now = datetime.datetime.now()
            subpath = "results_{}".format(now.strftime("%Y%m"))
            name = "{}_{}".format(now.strftime("%d%H%M"), name)
            self.__save_path__ = os.path.abspath(os.path.join(self.base_path, subpath, name))
        else:
            self.__save_path__ = os.path.abspath(os.path.join(self.base_path, name))
        return self.__save_path__

    def get_save_path(self, file="", no_sub_folders=False):
        if no_sub_folders:
            folder_path = self.__save_path__
        else:
            folder_path = os.path.join(self.__save_path__, self.file_tag)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        return os.path.join(folder_path, file)

    def get_best_model_path(self, stage_name):
        file_name = "{}_{}".format(stage_name, BaseSolver.best_model_name)
        return self.get_save_path(file_name)

    def get_recent_model_path(self, stage_name):
        file_name = "{}_{}".format(stage_name, BaseSolver.recent_model_name)
        return self.get_save_path(file_name)

    def __patch_data__(self, model, data):
        """patch data from a batch

        Args:
            data (Tensor): data of a batch

        Returns:
            _type_: _description_
        """
        x, y = data
        x = self.to_device(x)
        y = self.to_device(y)
        return x, y

    def __back_patch__(self, model_output, ground_truth):
        """pack the model output and ground truth to a dictionary, the key will be use in the loss function and metric

        Args:
            model_output (any): contains everything of the model output
            ground_truth (any): contains everything of the groundtruth

        Returns:
            Tuple|Dictionary: packed results. The default value is tuple of (model_output, ground_truth)
        """
        return (model_output, ground_truth)

    # @torch.compile
    def __forward__(self, model, x) -> torch.Tensor:
        if isinstance(x, tuple):
            y = model(*x)
        else:
            y = model(x)
        return y

    def run_epoch(
        self, stage_name, model, data, criteria: Caculator, loss_func: Caculator, optimizer=None, scheduler=None
    ):
        criteria.init_data()
        if optimizer is None:
            model.eval()
        else:
            loss_func.init_data()
            model.train()
        ep_id = self.data.get_counter(stage_name)
        bar_desc = f"Ep[{ep_id}] {stage_name}"
        # init progress bar
        dataset = tqdm(data, desc=bar_desc, total=len(data), bar_format=BAR_FMT)
        for data in dataset:
            model_input, ground_truth = self.__patch_data__(model, data)
            # forward
            self.reporter.start_timer("forward")
            if optimizer is not None:
                model_output = self.__forward__(model, model_input)
            else:
                with torch.no_grad():
                    model_output = self.__forward__(model, model_input)
            self.reporter.end_timer("forward")
            # backward
            self.reporter.start_timer("backward")
            backward_data = self.__back_patch__(model_output, ground_truth)
            try:
                # cal criteria
                with torch.no_grad():
                    criteria.cal(backward_data)
                if optimizer is not None:
                    loss = loss_func.cal(backward_data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            except NanOrInfException:
                pass
            self.reporter.end_timer("backward")
            # show progress
            dataset.set_postfix_str(f"{loss_func.str_results()}|{ criteria.str_results()}")
        self.data.new_datas(stage_name, loss_func.get_results() + criteria.get_results())
        # update learning rate
        if scheduler is not None:
            scheduler.step()

    def start(self, stage_list: list[str], tag, init_model=True, init_seed=0):
        self.file_tag = tag
        # init all models
        for model_name, model_info in self.models.items():
            model = model_info["model"]
            # output model info
            self.reporter.set_value(model_name, model, TAG_MOD)
            if init_model:
                model = model_info["initor"].init_model(model)
        # init optimizer and scheduler
        for stage_name, stage_info in self.stages.items():
            if "optimizer" in stage_info:
                stage_info["optimizer"].load_state_dict(stage_info["optimizer_init"])
            if "scheduler" in stage_info:
                stage_info["scheduler"].load_state_dict(stage_info["scheduler_init"])
        self.reporter.init()
        self.data.clear()

        # start
        self.set_seed(init_seed, "running start")
        self.reporter.start_timer("total")
        best_ep = 0
        self.msg(f"==========={self.file_tag} start==========", logging.INFO, True, True)
        for stage_name in stage_list:
            stage_info = self.stages[stage_name]
            stage_type = stage_info["type"]
            # call function
            if stage_type == "callback":
                stage_info["method"](*stage_info["args"])
                self.msg(f"call func {stage_name}.", lazy_mode=True)
                continue
            # load best model parameters from target stage
            elif stage_type == "load_state":
                model = self.models[stage_info["model_name"]]["model"]
                # load best model
                load_model_param_from = stage_info.get("target_stage", None)
                if load_model_param_from is not None:
                    best_model_path = self.get_best_model_path(load_model_param_from)
                    model = load_model_state(model, best_model_path)
                self.msg(
                    f"{stage_info['model_name']} load parameters from stage: {stage_info['target_stage']}.",
                    lazy_mode=True,
                )
                continue
            # training and evaluation
            model = self.models[stage_info["model_name"]]["model"]
            initor: ModelInitor = self.models[stage_info["model_name"]]["initor"]
            criterias = stage_info["metric"]
            data = stage_info["data"]
            if stage_type == "train":
                optim = stage_info["optimizer"]
                scheduler = stage_info["scheduler"]
                loss_func = stage_info["loss"]
            elif stage_type == "eval":
                optim = scheduler = None
                loss_func = Caculator(None)

            ep_id = self.data.get_counter(stage_name)
            model = self.to_device(model)
            self.run_epoch(stage_name, model, data, criterias, loss_func, optim, scheduler)
            for new_id, f_name, group, new_value in self.data.get_new_results(stage_name):
                self.reporter.set_results(stage_name, f_name, new_value, group)
            self.generate_report(temp=True)
            potential_best_ep = self.data.get_best(stage_name)
            best_ep = potential_best_ep["summary"]
            self.data.plot(stage_name, self.get_save_path(f"{stage_name}.png"))
            self.__save__(model, self.get_recent_model_path(stage_name))
            # save model if better performance obtained
            result_str = f"{stage_name}-ep[{ep_id}] results: {criterias.str_results()} || {loss_func.str_results()}"
            if best_ep - 1 == ep_id:
                result_str += " (best)"
                self.__save__(model, self.get_best_model_path(stage_name))
            self.msg(result_str, logging.DEBUG)
            self.data.save(self.get_save_path("scores.csv"))
        # save data
        self.reporter.end_timer("total")
        self.msg(f"{NAME_TRAIN} done.", logging.DEBUG)
        self.generate_report()
        self.msg(f"figure and model state will be save to {self.get_save_path()}")

    def generate_report(self, addition_info="", temp=False):
        self.reporter.flush(self.__save_path__, f"{self.file_tag}|{addition_info}", temp)
        if not temp:
            print(f"report generated to: {self.__save_path__}")

    def add_model(self, model_name, model, init_method=None, init_seed=None, init_path=None, init_filter=None):
        model_initor = ModelInitor()
        init_state = copy.deepcopy(model.cpu().state_dict())
        model_initor.set_init_param("init_param", init_state, 0)
        if init_path is not None:
            model_initor.set_init_path("init_path", init_path, 10, init_filter)
        if init_method is not None:
            model_initor.set_init_method("init_method", init_method, 10)
        self.models[model_name] = {"model": model, "initor": model_initor}

    def set_model_init_seed_all(self, seed):
        for model_info in self.models.values():
            model_info["init_seed"] = seed

    def add_stage(self, name, model_name, data, criterias, loss_func, back_param=None, **kws):
        stage_info = {"type": "train", "model_name": model_name, "data": data, **kws}
        if back_param is None:
            back_param = self.models[model_name]["model"].parameters()
        optim = Adam(
            back_param,
            stage_info.get("lr", 1e-3),
            weight_decay=stage_info.get("weight_decay", 0),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, stage_info.get("scheduler_step", 10), stage_info.get("gamma", 0.5)
        )
        stage_info["optimizer_init"] = copy.deepcopy(optim.state_dict())
        stage_info["scheduler_init"] = copy.deepcopy(scheduler.state_dict())

        self.reporter.add_data(name, data)
        stage_info["metric"] = Caculator(criterias, "score", 1)
        stage_info["loss"] = Caculator(loss_func, "loss", 0)
        stage_info["optimizer"] = optim
        stage_info["scheduler"] = scheduler
        self.data.add_lower_better(*stage_info["loss"].funcs.keys())
        self.stages[name] = stage_info

    def add_load_stage(self, name, model_name, stage_name):
        stage_info = {"type": "load_state", "model_name": model_name, "target_stage": stage_name}
        self.stages[name] = stage_info

    def add_eval_stage(self, name, model_name, data, criterias):
        stage_info = {"type": "eval", "model_name": model_name, "data": data}
        self.reporter.add_data(name, data)
        stage_info["metric"] = Caculator(criterias, "score", 1)
        self.stages[name] = stage_info

    def add_callback_stage(self, name, method, *args):
        stage_info = {"type": "callback", "method": method, "args": args}
        self.stages[name] = stage_info
