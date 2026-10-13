from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F




# Rede Neural = Multi Layer Perceptron = Feed-Forward Network
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.num_embedding, 4 * config.num_embedding)
        # Gaussian Error Linear Unit (funcao de acao probabilistica)
        # GELU, ao contrario do ReLU, nao dispensa os valores negativos
        #'tanh'~ HyperbolicTangent funcao para aproximacao de valores, garante mais velocidade.
        self.gelu = nn.GELU(approximate='tanh') 

        self.c_proj = nn.Linear(4 * config.num_embedding, config.num_embedding)

    # 'forward' nao tem significado, apenas o nome dado para o caminho da pipeline.
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x




# Block = bloco de Decoder do Transformer.
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.num_embedding) # Camada de normalizacao 1
        self.attn = CausalSelfAttention(config) # atencao
        self.ln_2 = nn.LayerNorm(config.num_embedding) # Camada de normalizacao 2
        self.mlp = MLP(config) #multi layer perceptron (feed-forward network -> FFN)

    # 'forward' nao tem significado, apenas o nome dado para o caminho da pipeline.
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