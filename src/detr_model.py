import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
detr.train()