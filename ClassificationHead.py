import torch

class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super(ClassificationHead, self).__init__()
        dropout_rate = 0.1
        self.cls_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),  # added in book summaries
            torch.nn.Linear(hidden_size, 768),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 227)  # 227 classes for book summaries dataset
        )

    def forward(self, x):
        return self.cls_head(x)

