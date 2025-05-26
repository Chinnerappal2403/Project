import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        return text, target
    
    def __len__(self):
        return len(self.targets)

