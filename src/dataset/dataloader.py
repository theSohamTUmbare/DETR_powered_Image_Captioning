from torch.utils.data import Dataset, DataLoader
from config import config
from dataset.dataset import CLIP_COCO_dataset, clip_coco_collate_fn
import os.path as op
from dataset.dataset import clip_coco_dataset, test_clip_coco_dataset


train_loader = DataLoader(
    clip_coco_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=clip_coco_collate_fn
)


test_loader = DataLoader(
    test_clip_coco_dataset,
    batch_size=config.BATCH_SIZE*4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=clip_coco_collate_fn
)