from torch.utils.data.dataset import Dataset
import numpy as np
import torch


class ACPDataset(Dataset):
    def __init__(self, data=None, train=True):
        super(ACPDataset, self).__init__()
        if train:
            self.data = data[0]
            self.target = data[1]
        else:
            self.data = data[2]
            self.target = data[3]

        self.data = self.data[:, np.newaxis, :]
        self.data = torch.from_numpy(self.data)

    def __getitem__(self, item):
        return [self.data[item], self.target[item]]

    def __len__(self):
        return self.data.size()[0]


class TestDataset(Dataset):
    def __init__(self, data=None):
        super(TestDataset, self).__init__()
        if type(data) == np.ndarray:
            self.data = torch.tensor(data=data)
        else:
            self.data = torch.tensor(data=data.values)
        self.data = self.data[:, np.newaxis, :]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.size()[0]