import datetime
import time
import sys
import os


TAG_BASE = "base info"
TAG_CONFIG = "configuarion"
TAG_RESULT = "results"
TAG_DATA = "data"
TAG_SCH = "scheduler"
TAG_OPT = "optimizer"
TAG_MOD = "model structure"
TAG_RUNNING = "train and eval"
TAG_SCRIPTS = "execute scripts"

STD_LIST = [
    TAG_BASE,
    TAG_CONFIG,
    TAG_RESULT,
    TAG_RUNNING,
    TAG_DATA,
    TAG_OPT,
    TAG_SCH,
    TAG_MOD,
    TAG_SCRIPTS,
]


def time_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_class(c):
    c = str(c).split(" at ")[0].replace("<", "")
    return c


class SubModule:
    ignore_names = ["params"]

    def __init__(self, name):
        self.divider = f"{'='*100}\n{'||{:^96}||'}\n{'='*100}\n".format(name)
        self.attr = {}

    def set_item(self, name, value, **kws):
        if name not in SubModule.ignore_names:
            self.attr[name] = value

    def __str__(self) -> str:
        content = self.divider
        for k, v in self.attr.items():
            if len(k) > 0:
                content += f"{k:30} : {v}\n"
            else:
                content += f"{v}\n"
        return content


class ResultModule(SubModule):
    def __init__(self, name):
        super().__init__(name)
        self.criteria_names = {}

    def set_item(self, trail_name, criteria_name, criteria_score, group=0, **kws):
        if trail_name not in self.attr.keys():
            self.attr[trail_name] = {}
        if criteria_name not in self.criteria_names:
            names = self.criteria_names.get(f"group{group}", [])
            if criteria_name not in names:
                names.append(criteria_name)
            self.criteria_names[f"group{group}"] = names
        self.attr[trail_name][criteria_name] = criteria_score

    def __str__(self) -> str:
        content = self.divider
        content += f"|{'criteria':10}|"
        trail_names = list(self.attr.keys())
        criteria_names = []
        for names in self.criteria_names.values():
            criteria_names.extend(names)
        for trail_name in trail_names:
            content += f"{trail_name:10}|"
        for c_name in criteria_names:
            content += f"\n|{c_name:10}|"
            for trail_name in trail_names:
                try:
                    score = "{:10.4f}|".format(self.attr[trail_name][c_name])
                except KeyError:
                    score = f"{'-':10}|"
                content += score
        content += "\n"
        return content


class Reporter:
    def __init__(self) -> None:
        self.modules = {}
        self.__record_scripts__()
        self.init()

    def init(self):
        self.modules[TAG_RESULT] = ResultModule(TAG_RESULT)
        self.modules[TAG_RUNNING] = SubModule(TAG_RUNNING)
        self.timer = {}
        # basic info
        self.set_value("start time", time_now())

    def set_value(self, name, value, module=TAG_BASE):
        if module not in self.modules.keys():
            self.modules[module] = SubModule(module)
        self.modules[module].set_item(name, value)

    def set_values(self, module=TAG_BASE, prefix="", postfix="", **kws):
        if module not in self.modules.keys():
            self.modules[module] = SubModule(module)
        for k, v in kws.items():
            self.modules[module].set_item(k, f"{prefix} {v} {postfix}")

    def set_results(self, trail_name, criteria_name, score, group):
        self.modules[TAG_RESULT].set_item(trail_name, criteria_name, score, group)

    def flush(self, save_path, tag, temp=False):
        for name, v in self.timer.items():
            self.set_value(f"time for {name}", f"{v['total']:.2f}s", TAG_RUNNING)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{tag}|report.txt")
        temp_path = path.replace(".txt", "_temp.txt")
        # add temp postfix if this is a temp report
        if temp:
            path = temp_path
        # remove temp path if this is not a temp report
        elif os.path.exists(temp_path):
            os.remove(temp_path)
        output = []
        for name in STD_LIST:
            if name in self.modules.keys():
                output.append(self.modules[name])
        for k, v in self.modules.items():
            if k not in STD_LIST:
                if k.endswith("data"):
                    output.insert(4, v)
                else:
                    output.append(v)
        with open(path, "w", encoding="utf-8") as f:
            for module in output:
                f.write(str(module))
        return path

    def start_timer(self, name):
        if self.timer.keys().__contains__(name):
            self.timer[name]["start"] = time.time()
        else:
            self.timer[name] = {"total": 0, "start": time.time()}

    def end_timer(self, name):
        self.timer[name]["total"] += time.time() - self.timer[name]["start"]

    def setting(self, model=None, loss_func=None, scheduler=None, optimizer=None):
        if model:
            self.set_value("Model", model.__class__.__name__)
        if loss_func:
            self.set_value("Loss Function", str(loss_func))
        # scheduler
        if scheduler:
            self.set_value("Class Name", get_class(scheduler), TAG_SCH)
            self.set_values(module=TAG_SCH, **(scheduler.state_dict()))
        # optimizer
        if optimizer:
            self.set_value("Class Name", optimizer.__class__.__name__, TAG_OPT)
            for group in optimizer.param_groups:
                self.set_values(module=TAG_OPT, **group)
        # model
        if model:
            self.set_value("", model, TAG_MOD)

    def __record_scripts__(self):
        file_name = sys.argv[0]
        with open(file_name, "r", encoding="utf-8") as f:
            self.set_value("", f.read(), TAG_SCRIPTS)

    def add_data(self, name, data, **cfg):
        if data is None:
            return
        self.set_value(f"{name} data size", len(data.dataset), TAG_DATA)
        self.set_value(f"{name} data class", get_class(data.dataset), TAG_DATA)
        self.set_values(TAG_DATA, prefix=name, **cfg)
