import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset, DataLoader


class TstDataset(Dataset):
    def __init__(self):
        self.data = list(range(10))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        random.shuffle(self.data)


class TstDataloader(DataLoader):
    def __init__(self):
        self.dataset = TstDataset()
        super().__init__(self.dataset)

    def shuffle(self):
        self.dataset.shuffle()


if __name__ == '__main__':
    dataloader = TstDataloader()
    for i in range(5):
        for elel in dataloader:
            print(elel, end=' ')
        print()
        dataloader.shuffle()
