from torch.utils.data.dataset import Dataset


class DatasetOneLabel(Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return example, target

    def __len__(self):
        return len(self.dataset)
class DatasetTwoLabels(Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target1, target2 = self.dataset[index]
        return example, target1, target2

    def __len__(self):
        return len(self.dataset)



