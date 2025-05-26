import lightning as L
from MyDataSet import MyDataSet
import torch
from torch.utils.data import Dataset, DataLoader

def get_train_test_val_Loaders(tx_train, tx_test, tx_val, labels_train, labels_test, labels_val, BATCH_SIZE):
    # Create train dataset and dataloader
    train_dataset = MyDataSet(
        texts=tx_train,
        targets=labels_train
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,  # was 8
        shuffle=True,
        drop_last=True
    )
    
    # Create test dataset and dataloader
    test_dataset = MyDataSet(
        texts=tx_test,
        targets=labels_test
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # larger batch size causes memory error
        shuffle=False,
        drop_last=False
    )
    
    # Create validation dataset and dataloader
    val_dataset = MyDataSet(
        texts=tx_val,
        targets=labels_val
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # was 16
        shuffle=False,
        drop_last=False
    )
    
    return train_dataloader, test_dataloader, val_dataloader


def tokenize_text(text, tokenizer):
    """
    Tokenize the text and return PyTorch tensors with dynamic padding
    """
    encodings = tokenizer(
        text,
        return_tensors='pt',
        padding='longest',  # Dynamically pad each batch to the length of the longest sequence
        add_special_tokens=False
    )
    return encodings


def get_optimizer(model, learning_rate=0.0001, diff_lr=0.00001, weight_decay=0.01):
    """
    Get optimizer with different learning rates for specified layers.
    Args:
        model (torch.nn.Module): The neural network model.
        learning_rate (float): Learning rate for non-differential layers.
        diff_lr (float): Learning rate for differential layers.
        weight_decay (float): Weight decay (decoupled from L2 penalty) for optimizer.
    Returns:
        torch.optim.AdamW: Optimizer for the model.
    """
    # Define parameters with different learning rates and weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    differential_layers = ['llm']
    
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": diff_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": diff_lr,
                "weight_decay": 0,
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    return optimizer

