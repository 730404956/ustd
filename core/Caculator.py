from torch import Tensor
import random
import torch


class NanOrInfException(Exception):
    def __init__(self, *args: object) -> None:
        self.str = "".join(args)

    def __str__(self) -> str:
        return self.str


class Caculator:
    def __init__(self, funcs: list[dict], sum_name="score", group=None) -> None:
        """_summary_

        Args:
            indicators (list): _description_
            funcs (list): list of {"name":str,"func":callable,"weight":float}
        """
        self.sum_name = sum_name
        self.funcs = {}
        if funcs is not None:
            for func in funcs:
                if group is not None:
                    func.update(group=group)
                self.add_func(**func)
        self.init_data()

    def add_func(self, name="no_name", group=0, func=None, weight=1):
        """添加一个计算函数

        Args:
            name (str, optional): 函数显示名称. Defaults to "no_name".
            func (func, optional): 实际调用的方法. Defaults to None.
            weight (float, optional): 计算所有函数值的和时的权重. Defaults to 1.
        """
        self.funcs[name] = {
            "func": func,
            "weight": weight,
            "group": group,
            "counter": 0,
            "temp": 0,
        }

    def init_data(self):
        for name in self.funcs.keys():
            self.funcs[name]["counter"] = 0
            self.funcs[name]["temp"] = 0

    def is_empty(self):
        return len(self.funcs) == 0

    def cal(self, *inputs) -> Tensor:
        total = 0
        for f_name, func in self.funcs.items():
            result = func["func"](*inputs)
            if isinstance(result, torch.Tensor):
                if torch.any(torch.isnan(result)) or torch.any(result.abs() == torch.inf):
                    raise NanOrInfException(f"func [{f_name}]:{result}")
            if isinstance(result, Tensor):
                result = result.mean()
            total += result * func["weight"]
            if isinstance(result, Tensor):
                result = result.item()
            func["counter"] += 1
            func["temp"] += result
        return total

    def get_results(self):
        results = []
        for func_name, func in self.funcs.items():
            _, _, group, counter, value = func.values()
            if counter == 0:
                mean_value = 0
            else:
                mean_value = value / counter
            results.append((func_name, group, mean_value))
        return results

    def str_results(self) -> Tensor:
        results_str = []
        for func_name, _, value in self.get_results():
            results_str.append(f"{func_name}={value:.3f}")
        return ",".join(results_str)

    def __str__(self) -> str:
        res_str = ["List of func"]
        for f_name, func in self.funcs.items():
            name = str(func["func"]).split(" at ")[0].replace("<", "")
            res_str.append(f"    weight:{func['weight']:4}||name:{f_name:6}||method:{name}")
        return "\n".join(res_str)


if __name__ == "__main__":

    def func(a):
        return 10

    def func2(a):
        return random.random()

    funcs = [
        {"name": "f1", "func": func},
        {"name": "f2", "func": func2},
    ]
    cc = Caculator(funcs, sum_type=Caculator.SCORE_TYPE_SUM)
    for i in range(6):
        cc.init_data()
        for j in range(8):
            cc.cal(j)
        print(cc.str_results())
