import torch


def warp(func, *dims):
    if func is None:
        return EmptyWarper(f"value at {dims}", dims)
    elif isinstance(dims[0], (str, int)):
        return SingleFuncWarper(func, dims)
    else:
        return MultiFuncWarper(func, dims)


class EmptyWarper(torch.nn.Module):
    def __init__(self, func, dims) -> None:
        super().__init__()
        self.dims = dims
        self.func = func

    def __str__(self) -> str:
        return str(self.func)

    def forward(self, inputs: dict):
        result = 0
        for dim in self.dims:
            result += inputs[dim]
        return result / len(self.dims)


class MultiFuncWarper(EmptyWarper):
    def forward(self, inputs: dict):
        result = 0
        for dim in self.dims:
            data = [inputs[d] for d in dim]
            result += self.func(*data)
        return result / len(self.dims)


class SingleFuncWarper(EmptyWarper):
    def forward(self, inputs: dict):
        result = 0
        for dim in self.dims:
            result += self.func(inputs[dim])
        return result / len(self.dims)


if __name__ == "__main__":

    def func(a, b):
        return a + b

    def func2(x):
        return x

    new_func1 = warp(None, "a1", "b1")
    new_func2 = warp(func2, "a1")
    new_func2 = warp(func2, "a1", "b2", "b1")
    new_func = warp(func, ["a1", "b1"], ["a2", "b2"])
    data = {"a1": 1, "b1": 2, "a2": 3, "b2": 4}
    print(new_func(data))
    print(new_func1(data))
    print(new_func2(data))
