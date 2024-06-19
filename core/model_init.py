from collections import OrderedDict
import torch.nn as nn
import torch


def weight_init_default(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.RNNBase):
        for layer_p in m._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.kaiming_normal_(m.__getattr__(p), mode="fan_out", nonlinearity="relu")
                elif "bias" in p:
                    nn.init.normal_(m.__getattr__(p))
    else:
        try:
            m.reset_parameters()
        except:
            pass


def load_model_state(model, state_path, filters=None, rename_leys=None):
    device = next(model.parameters()).device
    stat: OrderedDict = torch.load(state_path, map_location=device)
    if filters is not None:
        if isinstance(filters, str):
            stat = {k: v for k, v in stat.items() if filters in k}
        else:
            new_stat = {}
            for k, v in stat.items():
                for filter in filters:
                    if filter in k:
                        new_stat[k] = v
                        break
            stat = new_stat
        assert len(stat) > 0
    if rename_leys is not None:
        for src_name, new_name in rename_leys:
            stat = {k.replace(src_name, new_name): v for k, v in stat.items()}
        assert len(stat) > 0
    if isinstance(model, nn.DataParallel):
        src_dict = model.module.state_dict()
        src_dict.update(stat)
        imp_keys = model.module.load_state_dict(stat, strict=False)
    else:
        src_dict = model.state_dict()
        src_dict.update(stat)
        imp_keys = model.load_state_dict(src_dict, strict=False)
    print(imp_keys)
    return model


class ModelInitor:
    def __init__(self) -> None:
        self.init_methods = {}

    def reactivate(self):
        for v in self.init_methods.values():
            v["rest_init"] = v["init_limit"]

    def set_init_path(self, name, path: str, prior: int, filters=None, map_name=None, limit=1):
        self.init_methods[name] = {
            "rest_init": limit,
            "init_limit": limit,
            "prior": prior,
            "init_method": "from_path",
            "init_path": path,
            "filters": filters,
            "map_name": map_name,
        }

    def set_init_param(self, name, state_dict, prior: int, limit=1):
        self.init_methods[name] = {
            "rest_init": limit,
            "init_limit": limit,
            "prior": prior,
            "init_method": "from_dict",
            "state_dict": state_dict,
        }

    def set_init_method(self, name, init_method, prior: int, limit=1):
        self.init_methods[name] = {
            "rest_init": limit,
            "init_limit": limit,
            "prior": prior,
            "init_method": "from_random",
            "apply_method": init_method,
        }

    def init_model(self, model: nn.Module):
        init_methods = sorted(self.init_methods.items(), key=lambda x: x[-1]["prior"],reverse=True)
        for name, init_info in init_methods:
            if init_info["rest_init"] <= 0:
                continue
            init_info["rest_init"] -= 1
            if init_info["init_method"] == "from_path":
                model = load_model_state(
                    model, init_info["init_path"], init_info["filters"], init_info["map_name"]
                )
            elif init_info["init_method"] == "from_dict":
                src_dict = model.state_dict()
                src_dict.update(init_info["state_dict"])
                model.load_state_dict(src_dict, strict=False)
            elif init_info["init_method"] == "from_random":
                model.apply(init_info["apply_method"])
            else:
                pass
            break
        else:
            model.apply(weight_init_default)
        return model
