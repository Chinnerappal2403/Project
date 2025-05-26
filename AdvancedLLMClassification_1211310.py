import sys
import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import os
import time
import zipfile
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
import lightning as L
import dataloader
import Utils
from ClassificationNetLoRA import ClassificationNetLoRA

#MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # microsoft/Phi-3-mini-4k-instruct
MODEL_NAME = "h2oai/h2o-danube-1.8b-chat"
DO_TEST = False  # set to false for training, true for testing
BATCH_SIZE = 1
CLASSIFICATION_TYPE = 'MULTI_LABEL'  # or MULTI_CLASS, some datasets have multiple labels
GRADIENT_ACCUMULATION_STEPS = 4  # will make effective batch size = 4, if BATCH_SIZE=1
num_epochs = 5
seed = 252  # for reproducability
APPLY_LORA = True  # if false, then only classification head is fine tuned,
MAX_SEQ_LENGTH = 768  # embed size is 2560 # MAX_SEQ_LENGTH is context length
# H2o and Phi3 can handle 4k tokens
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def calc_accuracy(dataloader, model, tokenizer, type):  # for binary or multi-class
    with torch.no_grad():
        model.eval()
        pred_scores = []
        actual_scores = []
        max_token_len = 0
        wrong_count = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer)
            input_ids = encodings['input_ids'].to(device)
            if input_ids.shape[1] > max_token_len:
                max_token_len = input_ids.shape[1]
            attention_mask = encodings['attention_mask'].to(device)
            # truncate inputs, no division into segments
            if input_ids.shape[1] >= MAX_SEQ_LENGTH:
                input_ids = input_ids[:, 0:MAX_SEQ_LENGTH - 1]
            if attention_mask.shape[1] >= MAX_SEQ_LENGTH:
                attention_mask = attention_mask[:, 0:MAX_SEQ_LENGTH - 1]
            with autocast():  # handles mixed precision arithmetic automatically
                logits = model(input_ids, attention_mask)
                pred_score = F.softmax(logits, dim=-1).argmax(dim=-1).cpu().detach().numpy().tolist()
                pred_scores.extend(pred_score)
                actual_scores.extend(targets.numpy().tolist())
                if (pred_score[0] != targets[0]):
                    wrong_count = wrong_count + 1
                    # print('mismatch found : ',' actual=',targets[0],'pred=',pred_score[0], ' length=', len(encodings['input_ids'][0]), ' wrong count=',wrong_count)
        pred_scores = np.array(pred_scores)
        accuracy = accuracy_score(actual_scores, pred_scores)
        return accuracy


def calc_accuracy_multi_label(dataloader, model, tokenizer, type):  # for multi-label
    with torch.no_grad():
        model.eval()
        labels = []
        predictions = []
        pred_scores = []
        actual_scores = []
        max_token_len = 0
        wrong_count = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer)
            input_ids = encodings['input_ids'].to(device)
            if input_ids.shape[1] > max_token_len:
                max_token_len = input_ids.shape[1]
            attention_mask = encodings['attention_mask'].to(device)
            # truncate inputs, no division into segments
            if input_ids.shape[1] >= MAX_SEQ_LENGTH:
                input_ids = input_ids[:, 0:MAX_SEQ_LENGTH - 1]
            if attention_mask.shape[1] >= MAX_SEQ_LENGTH:
                attention_mask = attention_mask[:, 0:MAX_SEQ_LENGTH - 1]
            with autocast():  # handles mixed precision arithmetic automatically
                logits = model(input_ids, attention_mask)
                preds = torch.sigmoid(logits).detach().cpu()
                predictions.extend(preds)
                for out_labels in targets.detach().cpu():
                    labels.append(out_labels)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        y_preds = predictions.numpy()
        y_true = labels.numpy()
        y_pred_labels = np.where(y_preds > 0.5, 1, 0).tolist()
        accuracy = f1_score(y_true, y_pred_labels, average='micro')
        return accuracy


def main():
    max_seq_length = 1024
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              max_seq_length=max_seq_length,
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # ------------prepare data loaders--------------------------
    text_set, vectorized_labels, num_labels = dataloader.prepare_book_summaries(False)
    tx_train = [tokenizer.bos_token + x for x in text_set["train"]]
    tx_test = [tokenizer.bos_token + x for x in text_set["test"]]
    tx_val = [tokenizer.bos_token + x for x in text_set["dev"]]
    if CLASSIFICATION_TYPE == 'MULTI_LABEL':  # book summaries and eurlex are multi-label
        labels_train = list(vectorized_labels["train"])
        labels_test = list(vectorized_labels["test"])
        labels_val = list(vectorized_labels["dev"])
    else:
        labels_train = list(vectorized_labels["train"].reshape(-1))
        labels_test = list(vectorized_labels["test"].reshape(-1))
        labels_val = list(vectorized_labels["dev"].reshape(-1))
    train_dataloader, test_dataloader, val_dataloader = Utils.get_train_test_val_Loaders(tx_train, tx_test, tx_val,
                                                                                         labels_train, labels_test, labels_val, BATCH_SIZE)
    sample = tokenizer(tx_train[0], add_special_tokens=False).input_ids
    decoded = tokenizer.decode(sample)
    # print(decoded)
    learning_rate = 0.0002
    diff_lr = 0.00001  # not being used as we freeze the llm backbone
    warmup_steps = 0
    weight_decay = 0.01
    # Set seed for reproducibility
    L.seed_everything(seed=seed)
    # create model
    model = ClassificationNetLoRA(MODEL_NAME, DO_TEST, APPLY_LORA)
    model.to(device)  # select device
    if DO_TEST == True:
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test')
        else:
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
        # val_acc = calc_accuracy(val_dataloader, model, tokenizer, type='val')
        # print('Train accuracy:', train_acc)
        print('Test accuracy:', test_acc)
        return

    # print names of trainable parameters
    print('Here are the trainable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    # Get the optimizer
    optimizer = Utils.get_optimizer(model,
                                    learning_rate=learning_rate,
                                    diff_lr=diff_lr,
                                    weight_decay=weight_decay
                                    )
    # Set up the scheduler for learning rate adjustment
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_dataloader)
    )
    scaler = GradScaler()
    optimizer.zero_grad()
    start_time = time.time()
    for epoch in range(num_epochs):
        max_token_len = 0
        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            prompt, targets = batch
            encodings = Utils.tokenize_text(prompt, tokenizer)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            if input_ids.shape[1] > max_token_len:
                max_token_len = input_ids.shape[1]
            if input_ids.shape[1] >= MAX_SEQ_LENGTH:  # truncate input
                input_ids = input_ids[:, 0:MAX_SEQ_LENGTH - 1]
            if attention_mask.shape[1] >= MAX_SEQ_LENGTH:
                attention_mask = attention_mask[:, 0:MAX_SEQ_LENGTH - 1]
            targets = targets.to(device)
            # forward pass with autocast for mixed precision training
            with autocast():
                logits = model(input_ids, attention_mask)
                if CLASSIFICATION_TYPE == 'MULTI_LABEL':
                    loss = F.binary_cross_entropy_with_logits(logits,
                                                             targets.float())  # sigmoid + binary cross entropy loss
                else:
                    loss = F.cross_entropy(logits, targets)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            # backward pass, optimization step, and learning rate adjustment
            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            # Logging training progress
            if (batch_idx % 100) == 0:
                print(
                    f'Epoch: {epoch + 1} / {num_epochs}'
                    f'| Batch: {batch_idx + 1}/{len(train_dataloader)}'
                    f'| Loss: {loss.item():.4f}'
                )
            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_dataloader)):
                loss.detach()
                attention_mask.detach()
                del attention_mask
                del encodings
                del loss
                del logits
                del prompt
                del targets
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test')
        else:
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
        print('Epoch=', epoch, ' Test accuracy:', test_acc)
        print(max_token_len)
        model.save_peft_adapter()
        if CLASSIFICATION_TYPE == 'MULTI_LABEL':
            test_acc = calc_accuracy_multi_label(test_dataloader, model, tokenizer, type='test')
        else:
            # train_acc = calc_accuracy(train_dataloader, model, tokenizer, type='train')
            test_acc = calc_accuracy(test_dataloader, model, tokenizer, type='test')
        # val_acc = calc_accuracy(val_dataloader, model, tokenizer, type='val')
        # print('Train accuracy:', train_acc)
        print('Final Test accuracy:', test_acc)
        # print('Val accuracy:', val_acc)
        # Save trained model
        # model.save_peft_adapter()


if __name__ == "__main__":
    sys.exit(int(main() or 0))

