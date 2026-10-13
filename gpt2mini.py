from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math



#
#-=-=-=-=-=-=-  COMECO GPT-2  -=-=-=-=-=-=-=-=--=-=
#
# Seguindo convencoes de nome do HuggingFace para podermos
# reutilizar os pesos pre-setados por eles.
# nao existe a necessidade de testarmos os embeddings de pesos
# alguem ja tentou as variacoes e chegou nesses numeros.


# Em cada camada de atenção, tem várias 'cabeças' funcionando em paralelo
# a saída de cada cabeça é concatenada umas com a outras em uma única
# gerando o output da camada de Multi-Head Attention.
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

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
        # B ~ tamanho do batch
        # T ~ comprimento da sequencia
        # C ~ dimensionalidade do embedding (n_embd)
        B, T, C = x.size()
        
        # calcula os valores de query(q), key(k), value(v) para todas as cabeças e move para o proximo batch
        # nh é 'number of heads', hs é 'head size' e C número de canais = nh * ns
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
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # mascara causal
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



#tamanho e estrutura do modelo
@dataclass
class GPTconfigs:
    block_size:int = 1024 # comprimento máx da sequência, em tokens, que o modelo pode processar em uma passagem (Janela de Contexto)|ou seja,0 a 1023 posicoes;1024x1024 mascara causal.
    vocab_size:int = 50257 # maximo numero de tokens do vocab/dicionario | 50.000 combinacoes do BPE + 256 bytes de tokens(para as folhas da arvore do BPE) + 1 <|endoftext|> token
    n_layer:int = 12 # numero de layers | quantos transformers serao empilhados.
    n_head:int = 12 # numero de cabecas de atencao em cada bloco que rodam em paralelo.
    n_embd:int = 768 # vetor que representa cada token ao longo dos blocos; 768 números por token.
    #
    #Assim, com os padrões atuais, o modelo processa sequências de até 1024 tokens, representados por vetores de 768 dimensões, 
    # passando por 12 blocos, cada um com 12 cabeças de atenção.



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
            # `hidden` -> layers, apos passar pela  (normalização1 e multihead-attn) -> h1
            # h1 passa pela (normalização2 e FeedForwardNetwork) -> h2; ou seja:
            # h(hidden) é a saída do Decoder antes da normalização e da projeção Linear + Softmax.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd), #normalizador
        ))
        # classificador final, projeta 768 dimensoes para o vocabulario de 50257
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    # pre-trained eh um construtor que retorna um objeto GPT() dado um `model_type`
    # nesse caso `model_type = gpt2` (que eh o gpt2 mini, o menor)
    #
    # carregar os parametros do HuggingFace/GPT-2, seria desnecessário não fazer isso
    # é apenas um trabalho massivo de teste e reteste até achar os números corretos.
    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        
        # carrega os pesos do modelo pretreinado do GPT-2/HuggingFace
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("carregando pesos do gpt pre-treinado: %s" % model_type)

        #n_layer, n_head, n_embd são determinados do model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),         #  124M  parametros
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), #  350M  parametros
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  #  774M  parametros
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)      # 1558M  parametros
        }[model_type]

        config_args['vocab_size'] = 50257 # Sempre 50257 para checkpoints do modelo do GPT
        config_args['block_size'] = 1024 # Sempre 1025 para checkpoints do modelo do GPT

        #inicializador do modelo do GPT2 'mini'
        config = GPTconfigs(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ignorando essa mascara/buffer que eh usada apenas na mascara autoregressiva.


        #incializador do modelo do HuggingFace/Transformer
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        #copiar tendo certeza que todos os parametros estao alinhados e correspondem em `name` e `shape`
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignorando essa mascara/buffer que eh usada apenas na mascara autoregressiva.
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #mesma coisa.

        # os checkpoints da OpenAI usam o modulo `Conv1D`, mas apenas queremos usar
        # vindos do repositorio do tensorflow, alguns pesos estao transpostos do que precisamos
        # para importamos para o PyTorch. entao vamos transpor-los de volta.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # tratamento para os pesos Conv1D que precisamos transpor
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copia normal dos parametros
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#-=-=-=-=-=-=-  FIM GPT-2  -=-=-=-=-=-=-=-=--=-=


model = GPT.from_pretrained('gpt2')
print("\n\t!!! nao crashou !!!")