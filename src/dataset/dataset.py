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

train_img_dir = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
train_annotation_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
val_img_dir = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
val_annotation_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json'


def read_json(fname):
    """Read a JSON file and return its contents as a dictionary.

    Args:
        fname (str or Path): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name

    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []

        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)

    return img_id_to_captions



def _transform():
    return Compose([
        Resize((800, 800)),          # Resize to 800
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""
    def __init__(self, text_tokenizer, img_dir=train_img_dir, the_annotation_file=train_annotation_file, context_length=config.CONTEXT_LEN):

        super(CLIP_COCO_dataset, self).__init__()

        annotation_file = the_annotation_file
        # print("annotation_file : ", annotation_file)
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = img_dir
        # print("img dir : ", self.img_dir)

        self.transform = _transform()
        self.context_length = context_length

        self.tokenizer = text_tokenizer
        self.tokenizer.add_special_tokens({ "pad_token": "[PAD]" })



    def tokenize(self, text):
        tokens = self.tokenizer(
            text,
            max_length=self.context_length,
            padding="do_not_pad",
            truncation=True,
            return_tensors="pt"
        )
        return tokens["input_ids"].squeeze()  # shape: (context_length,)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        all_caps = self.img_id_to_captions[img_id]
        # randomly picking one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        img_path = op.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenize(text)

        tokenized_refs = [self.tokenize(cap) for cap in all_caps]

        return img_input, text_input, tokenized_refs



clip_coco_dataset = CLIP_COCO_dataset(
    # hf_tokenizer,
    CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
)

test_clip_coco_dataset = CLIP_COCO_dataset(
    CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32'),
    img_dir=val_img_dir,
    the_annotation_file = val_annotation_file,
)

def clip_coco_collate_fn(batch):
    imgs, texts, tokenized_refs = zip(*batch)  

    # [B, C, H, W]
    imgs = torch.stack(imgs)

    # [B, T_text]
    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=clip_coco_dataset.tokenizer.pad_token_id
    )

    # finding max number of refs in this batch
    B = len(tokenized_refs)
    max_refs = max(len(refs) for refs in tokenized_refs)

    # for each sample, pad its list of refs up to max_refs
    padded_per_sample = []
    for refs in tokenized_refs:
        refs = list(refs)
        # pading missing captions with a single pad‑token sequence
        while len(refs) < max_refs:
            refs.append(
                torch.tensor([clip_coco_dataset.tokenizer.pad_token_id],
                             dtype=torch.long)
            )
        padded_per_sample.extend(refs)

    # now pading all sequences across the “flattened” list
    padded_flat = pad_sequence(
        padded_per_sample,
        batch_first=True,
        padding_value=clip_coco_dataset.tokenizer.pad_token_id
    )  # shape: [B * max_refs, L]

    # reshaping back [B, max_refs, L]
    L = padded_flat.size(1)
    tokenized_refs = padded_flat.view(B, max_refs, L)

    return imgs, texts, tokenized_refs  # [B, C, H, W], [B, T_text], [B, max_refs, L]


