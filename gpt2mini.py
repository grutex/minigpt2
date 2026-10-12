from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# Block 
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.num_embedding) # Camada de normalizacao 1
        self.attn = CausalSelfAttention(config) # atencao
        self.ln_2 = nn.LayerNorm(config.num_embedding) # Camada de normalizacao 2
        self.mlp = MLP(config) #multi layer perceptron (feed-forward network -> FFN)

    def forward(self, x):
        x += self.attn(self.ln_1(x)) # x vai para normalizacao 1 e depois atencao
        x += self.mlp(self.ln_2(x)) # x vai para normalizacao 2 e depois feed-forward
        return x # valor 


@dataclass
class GPTconfigs:
    block_size:int = 256
    vocab_size:int = 65
    num_layer:int = 6
    num_head:int = 6
    num_embedding:int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Abaixo, tentativa de dar match no transformer do hugging faces
        # Estrutura encontrada no arquivo `play.ipynb`

        # nn.Embedding - modulo wrapper de um unico array de numeros
        # permite acessar os elementos indexando-os em linhas.
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.num_embedding), #weigth tkn embd
            wpe = nn.Embedding(config.block_size, config.num_embedding), #weigth pos embd
            h = nn.ModuleList([Block(config) for _ in range(config.num_layer)]), #`hidden` -> layers
            ln_f = nn.LayerNorm(config.num_embedding), #normalizador
        ))
        # classificador final, projeta 768 dimensoes para o vocabulario de 50257
        self.lm_head = nn.Linear(config.num_embedding, config.vocab_size, bias=False)