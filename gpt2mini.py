from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTconfigs:
    block_size:int = 256
    vocabulary_size:int = 65
    num_layer:int = 6
    num_head:int = 6
    num_embedding:int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

