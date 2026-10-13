from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math



#
#-=-=-=-=-=-=-  COMECO GPT-2  -=-=-=-=-=-=-=-=--=-=
#
# Seguindo convencoes de nome do HuggingFaces para podermos
# reutilizar os pesos pre-setados por eles.


# Em cada camada de atenção, tem várias 'cabeças' funcionando em paralelo
# a saída de cada cabeça é concatenada umas com a outras em uma única
# gerando o output da camada de Multi-Head Attention.
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head

        #key, query, value para todas as cabeças; em batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        #output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        #regularização
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # nao é um viés de fato, uma máscara é a descrição melhor.
        # bias é uma convenção de nome da OpenAI.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    # 'forward' nao tem significado, apenas o nome dado para o caminho da pipeline.
    #
    # Cada token está alinhado em sequencia, e nesse caso existem 1020 deles
    # cada token nesse estágio da atenção emite 3 vetores: key(k), query(q), value(v)
    # querys e keys tem que multiplicar uns com os outros para ter o 'Attention Ammount', ou seja
    # o quão interessante eles acham uns aos outros.
    def forward(self, x):
        # B ~ tamanho do batch, ??
        # T ~ comprimento da sequencia ??
        # C ~ dimensionalidade do embedding (n_embd) ??
        B, T, C = x.size()
        
        # calcula os valores de query(q), key(k), value(v) para todas as cabeças e move para o proximo batch
        # nh é 'number of heads', ns é 'head size' e C número de canais = nh * ns
        #
        # GPT-2 mini(124M) : n_head=12 ; hs=64 ; logo, nh*ns=768 canais para o transformer.
        #
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=12)

        # B e nh sao tratados como batches, e por isso aplica as operações neles em paralelo
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # B, nh, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # B, nh, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # B, nh, T, hs

        # atençao, materializa a maior matriz transposta (T, T) para todas as querys(q) e keys(k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # mascara causal
        att = F.softmax(att, dim=-1) # softmax, normalizar a atencao, soma probabilistica igual a 1.

        # @ é para multiplicação de matrizes.
        # att @ v - é uma maneira de fazer uma soma com pesos, basicamente estamos somando a atencao
        # os tokens que achamos interessantes em cada um dos tokens('Attention Ammount')
        y = att @ v # (B, nh, T, ns) x (B, nh, T, hs) = (B, nh, T, hs)
        y = y.traspose(1, 2).contiguos().view(B, T, C) # remontar todos os outputs das cabeças, lado a lado, concatenacao.

        # projecao da saída.
        y = self.c_proj(y)
        return y


# Rede Neural = Multi Layer Perceptron = Feed-Forward Network
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Gaussian Error Linear Unit (funcao de acao probabilistica)
        # GELU, ao contrario do ReLU, nao dispensa os valores negativos
        #
        #'tanh'~ Hyperbolic Tangent funcao para aproximacao de valores, antigamente garantia mais velocidade.
        # hoje tanh é mais por saudosismo. o GELU pode ser usado diretamente.
        self.gelu = nn.GELU(approximate='tanh') 

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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
        self.ln_1 = nn.LayerNorm(config.n_embd) # Camada de normalizacao 1
        self.attn = CausalSelfAttention(config) # atencao
        self.ln_2 = nn.LayerNorm(config.n_embd) # Camada de normalizacao 2
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
    n_layer:int = 6
    n_head:int = 6
    n_embd:int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Abaixo, tentativa de dar match no transformer do hugging faces
        # Estrutura encontrada no arquivo `play.ipynb`

        # nn.Embedding - modulo wrapper de um unico array de numeros
        # permite acessar os elementos indexando-os em linhas.
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #weigth tkn embd
            wpe = nn.Embedding(config.block_size, config.n_embd), #weigth pos embd
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #`hidden` -> layers
            ln_f = nn.LayerNorm(config.n_embd), #normalizador
        ))
        # classificador final, projeta 768 dimensoes para o vocabulario de 50257
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#-=-=-=-=-=-=-  FIM GPT-2  -=-=-=-=-=-=-=-=--=-=