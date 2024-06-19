from ast import arg
from unittest import result
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import os
import multiprocessing
from multiprocessing import Manager
import time


def cache_worker(dataset: Dataset, save_pos, data_id, datas):
    data = dataset.__getitem__(data_id)
    data_path = os.path.join(save_pos, f"{data_id}.dat")
    torch.save(data, data_path)
    datas.append(data_path)


class CachedDataset(Dataset):
    def __init__(self, data, save_pos=None) -> None:
        super().__init__()
        self.data = []
        self.save_pos = save_pos
        self.get = None
        if isinstance(data, str):
            self.from_file(data)
        elif isinstance(data, Dataset):
            self.from_data(data)

    def from_data(self, dataset: Dataset, num_workder=3):
        if self.save_pos != None and isinstance(self.save_pos, str):
            if not os.path.exists(self.save_pos):
                os.makedirs(self.save_pos, exist_ok=True)
            else:
                # raise Exception(f"folder {self.save_pos} exist!")
                print("exist")
            manager = Manager()
            datas = manager.list()
            self.get = self.get_file_item
            pool = multiprocessing.Pool(num_workder)
            pbar = tqdm(total=len(dataset),desc="data preload")
            def update(a):
                pbar.update()
            for id in range(pbar.total):
                pool.apply_async(cache_worker, args=(dataset, self.save_pos, id, datas), callback=update)
            pool.close()
            pool.join()
            self.data.extend(datas)
        else:
            self.get = self.get_load_item
            for data_id in dataset:
                data = dataset.__getitem__(data_id)
                self.data.append(data)
        return self

    def from_file(self, path):
        if os.path.exists(path):
            if os.path.isfile(path):
                self.data = torch.load(path, map_location="cpu")
                self.get = self.get_load_item
            else:
                for f in os.listdir(path):
                    if f.endswith(".dat"):
                        self.data.append(os.path.join(path, f))
                self.get = self.get_file_item
            return self
        else:
            raise Exception(f"path {path} is invalid!")

    def save_data(self, file_path, mode="file"):
        if mode == "file":
            torch.save(self.data, file_path)
        elif mode == "folder":
            for i, dt in enumerate(self.data):
                torch.save(dt, os.path.join(file_path, f"{i}.dat"))

    def get_file_item(self, index: int):
        return torch.load(self.data[index], map_location="cpu")

    def get_load_item(self, index: int):
        return self.data[index]

    def __getitem__(self, index: int):
        if self.get:
            return self.get(index)
        else:
            raise Exception("data not loaded!")

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    pool = multiprocessing.Pool(2)
    '''
    for _ in tqdm(pool.imap_unordered(myfunc, range(100)), total=100):
        pass
    '''
    pbar = tqdm(total=10)
    results=[]
    def myfunc(a,b):
        time.sleep(0.5)
        return a
    def update(a):
        pbar.update()
        results.append(a)
        # tqdm.write(str(a))
    for i in range(pbar.total):
        pool.apply_async(myfunc, args=(i,i), callback=update)
    # tqdm.write('scheduled')
    pool.close()
    pool.join()
    print(results)
