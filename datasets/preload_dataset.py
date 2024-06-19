from torch.utils.data import Dataset
from tqdm import tqdm

class CachedDataset(Dataset):
    def __init__(self,dataset:Dataset) -> None:
        super().__init__()
        self.data=[]
        loop=tqdm(range(len(dataset)),total=len(dataset))
        for d in loop:
            loop.set_description("data preload")
            self.data.append(dataset.__getitem__(d))
        
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)
