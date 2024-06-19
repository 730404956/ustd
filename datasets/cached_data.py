from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os


def cache_data(data: Dataset, save_folder, batch_size, num_worker, collect_fn, shuffle, reload=True,pin_mem=True) -> DataLoader:
    if not os.path.exists(save_folder) or reload or abs(len(os.listdir(save_folder)) - len(data)) > len(data)*0.1:
        cached_dataset = PreloadDataset(data, save_folder)
        loader = DataLoader(cached_dataset, batch_size=batch_size, num_workers=num_worker, collate_fn=collect_fn, shuffle=shuffle)
        loop = tqdm(loader, total=len(loader), desc="data preload")
        for data in loop:
            pass
    datas = CachedDataset(save_folder)
    loader = DataLoader(datas, batch_size=batch_size, num_workers=num_worker, collate_fn=collect_fn, shuffle=shuffle,pin_memory=pin_mem)
    return loader


class DatasetBase(Dataset):
    def __init__(self, save_folder) -> None:
        super().__init__()
        if save_folder != None and isinstance(save_folder, str):
            self.save_folder = save_folder
        else:
            raise Exception(f"path not valid: {save_folder}")

    def get_path(self, index):
        return os.path.join(self.save_folder, f"batch_{index}.dat")


class PreloadDataset(DatasetBase):
    def __init__(self, dataset: Dataset, save_folder) -> None:
        super().__init__(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        path = self.get_path(index)
        torch.save(data, path)
        return data

    def __len__(self):
        return len(self.dataset)


class CachedDataset(DatasetBase):
    def __init__(self, save_folder) -> None:
        super().__init__(save_folder)
        self.data_path = os.listdir(self.save_folder)

    def __getitem__(self, index: int):
        path = self.get_path(index)
        return torch.load(path, map_location="cpu")

    def __len__(self):
        return len(self.data_path)
