import json
from collections import OrderedDict
from pathlib import Path
import os.path as op
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader  # Fixed import
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, Resize
from transformers import CLIPTokenizer
import os
import random
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from config import config

from dataset.dataloader import train_loader
from imgcap_model import ImgCapModel

    
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    start_epoch: int = 0,
    num_epochs: int = 1000,
    pad_token_id: int = config.PAD_ID,
    accumulation_steps: int =4,                    # <— new argument
) -> List[float]:
    """
    Train the captioning model with teacher forcing, using gradient accumulation.

    Args:
        model: your DETR+caption model.
        optimizer: optimizer for model parameters.
        train_loader: yields (images, token_ids) batches.
        start_epoch: which epoch to start from (for resuming).
        num_epochs: total number of epochs to run.
        pad_token_id: index to ignore in the loss.
        accumulation_steps: how many mini‐batches to accumulate before each optimizer.step().

    Returns:
        epoch_losses: list of average training loss per epoch (loss per token).
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    epoch_losses: List[float] = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        total_tokens = 0
        this_epoch_batch_losses: List[float] = []

        optimizer.zero_grad()  # zero out gradients at the very start of the epoch

        # get current LR (assumes single param_group)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n=== Epoch {epoch+1}/{num_epochs} | lr: {current_lr:.2e} ")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (images, captions, _) in enumerate(pbar):
            # Move to device
            images = images.to(config.DEVICE).float()
            captions = captions.to(config.DEVICE).long()

            # Forward pass (teacher forcing)
            # logits: [ B, T, V]
            logits = model(images, tgt_tokens=captions)

            # shift logits and targets for next-token prediction
            logits = logits[:, :-1, :]   # [B, T-1, V]
            targets = captions[:, 1:]    # [B, T-1]

            B, Tm1, V = logits.shape
            logits_flat  = logits.reshape(-1, V)     # [B*(T-1), V]
            targets_flat = targets.reshape(-1)       # [B*(T-1)]

            # compute “unscaled” loss (mean over all non-pad tokens in this mini-batch)
            loss_unscaled = loss_fn(logits_flat, targets_flat)
            # now scale it down so that after accumulation_steps of these, 
            # the gradient is effectively as if you used a single big batch.
            loss = loss_unscaled / accumulation_steps
            loss.backward()   # accumulate gradients

            # count how many real tokens (i.e. ignore pad) in this mini‐batch
            non_pad_tokens = targets_flat.ne(pad_token_id).sum().item()

            # For logging: we always want "true" loss_unscaled * num_tokens 
            # so that (running_loss / total_tokens) = avg loss‐per‐token
            running_loss += loss_unscaled.item() * non_pad_tokens
            total_tokens += non_pad_tokens

            this_epoch_batch_losses.append(loss_unscaled.item())

            # Only step & zero_grad every accumulation_steps mini‐batches,
            # or if it's the very last batch in the loader
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # update progress bar with the running avg loss per token
            avg_batch_loss = running_loss / total_tokens
            pbar.set_postfix(loss=avg_batch_loss)

        # at the end of the epoch, compute the average loss per token
        epoch_loss = running_loss / total_tokens
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch+1} completed — avg loss per token: {epoch_loss:.4f}")

        # save checkpoint every epoch (or adjust as needed)
        checkpoint_path = f"DETR_CAP{epoch+1:03d}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
        }, checkpoint_path)

    return epoch_losses


if __name__ == "__main__":

    model = ImgCapModel().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Start training
    train_losses = train(
        model,
        optimizer,
        train_loader,
        start_epoch=0,
        num_epochs=config.NUM_EPOCHS,
        pad_token_id=config.PAD_ID,
        accumulation_steps=config.ACCUMULATION_STEPS
    )
    
    print("Training completed. Losses per epoch:", train_losses)