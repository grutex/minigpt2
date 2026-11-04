from dataclasses import dataclass, asdict
import argparse
import hashlib
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import inspect

# Versão com quantidade de threads configurável por --threads.
#============================================================================
# A versão CPU do PyTorch é suficiente para executar este script.
# A execução deste script usa CPU em float32, com duas threads por padrão.
#
# OBS:
# Esse codigo estava crashando meu PC, provavelmente super aquecimento da CPU.
#
#
# Info:
# --threads: paralelismo da CPU (padrão: 2).
# --max-new-tokens: quantidade máxima de tokens novos (padrão: 40).
# --num-sequences: quantidade de textos gerados (padrão: 1).
#============================================================================

#Ugly x Beautiful Numbers
# Dimensões múltiplas de 128 podem favorecer kernels de GPU.
# 50304 é múltiplo de 128, não uma potência de dois. As posições extras
# do vocabulário devem ser excluídas da amostragem, pois não têm token no GPT-2.

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        # qkv tem shape (B, T, 3*C); divide a ultima dimensao em Q, K e V.
        q, k, v = qkv.split(self.n_embd, dim=-1)

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
        y = y.transpose(1, 2).contiguous().view(B, T, C) # remontar todos os outputs das cabeças, lado a lado, concatenacao.

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
        # A aproximação tanh mantém a variante de GELU usada pelo GPT-2.
        self.gelu = nn.GELU(approximate='tanh') 

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        x = x + self.attn(self.ln_1(x)) # x vai para normalizacao 1 e depois atencao
        x = x + self.mlp(self.ln_2(x)) # x vai para normalizacao 2 e depois feed-forward
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
        #
        # nn.Embedding - modulo wrapper de um unico array de numeros
        # permite acessar os elementos indexando-os em linhas.
        #
        # Quando criamos o nosso GPT.module, todas essas camadas e modulos
        #  já tem iniciadores aleatórios por padrão.
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

        # Weight Sharing Scheme / Weight Tying Scheme
        #
        # basicamente estamos apontando o wte.weight para o lm_head
        # ponteiro para lm_head. Copia a referencia, o data pointer.
        # nos deixando com um único Tensor que será usado 2 vezes
        # no Feed-Forward.
        #
        # No GPT-2 pequeno isso evita uma segunda matriz de 38.597.376
        # parâmetros: cerca de 124M compartilhados, em vez de 163M separados.
        #
        # Por causa do Weight Tying não precisamos treinar tantos parametros
        # tornando nosso modelo mais eficiente.
        #  -----------------------------------------------------------
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        #
        # apply:
        # itera todos os submodulos desse modulo
        # e aplica a função "_init_weights"
        self.apply(self._init_weights) 

    # Se o modulo é nn.Linear > inicie os pesos com uma normal com um
    # desvio-padrao de 0.02
    #
    # Se tiver um vies(bias) nessa camada > inicia com zero
    #
    # Se o modulo é nn.Embedding > inicie os pesos com uma normal com
    # desvio-padrao de 0.02
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # funcao que de fato vai encaminhar para geracao.
    # recebe : os indices de tokens (idx)
    # retorna:
    def forward(self, idx, targets=None): #Targets opicionais visto que agora temos como calcular o loss
        # idx tem o shape (B, T)
        # T nao pode ser maior que o block_size que eh o max do comprimento da sequencia. (1024)
        # cada linha dessa matriz pode ter ate `block_size` de tamanho.
        #
        # basicamente:
        # B - sequencias
        # T - maximo de tokens por sequencia
        B, T = idx.size()
        assert T <= self.config.block_size, f"Nao foi possivel encaminhar a sequencia de len:{T}, block size..."


        # encaminhamentos:
        #
        # encaminhamento dos embeddings de token e posicional
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # arange=range para o PyTorch | cria um indices de posicao.|device garantir que a GPU seja utilizada.
        pos_emb = self.transformer.wpe(pos) # embeddings posicionais de shape: (T, n_embd) | uma outra dimensao `fantasma` tem que ser criada para dar match no tok_emb.
        tok_emb = self.transformer.wte(idx) # embeddings de tokens de shape: (B, T, n_embd)
        x = tok_emb + pos_emb
        #
        # encaminhamento dos blocos para o transformer
        for block in self.transformer.h:
            x = block(x)
        #
        # encaminhamento da ultima camada de normalizacao e dos classificadores
        #
        # se o input foi BxT(B, T) indices, entao cada um dos BxT vai calcular
        # o `logits` para qual token deve ser o proximo na sequencia.
        # ou seja, o token BxT+1, o token logo a direita dele; o proximo.
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size) | vocab_size eh o numero de possiveis maximos tokens para serem o proximo
        loss = None #Default
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # O que acontece acima é que o cross_entropy não gosta de matrizes
            # multidimensionais então temos que passar por uma camada de achatamento
            # para tirarmos as 3D do Tensor e colocalo em 2D.
            #
            # irá virar um calculo BxT bidimensional (Linhas x 1) exemplos
            # individuais e multiplicas pelas vocab_size linhas.
            #
            # Também achatamos os target para virar um Tensor BxT (Linhas x 1)
            # então calculamos o Loss e retornamos.
        return logits, loss
# =============================================================================
# Logits são scores brutos(raw) e não normalizados produzidas por um modelo de
# aprendizado de máquina antes de serem convertidas em probabilidades por
# meio de uma função de ativação como softmax ou sigmoid.
#
# Esses logits estao apenas precisando passar pelo Softmax para
# virarem probabilidades.
# =============================================================================



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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # criar o otimizador AdamW e usar sua versão fundida se disponivel.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        extra_args = {'fused': use_fused} if fused_available else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **extra_args)
        return optimizer
#-=-=-=-=-=-=-  FIM GPT-2  -=-=-=-=-=-=-=-=--=-=


#=============  DADOS, TREINO E GERAÇÃO  =======================
BASE_DIR = Path(__file__).resolve().parent
# Cache local evita depender da pasta temporária do Windows para o tokenizer.
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(BASE_DIR / '.cache' / 'tiktoken'))
import tiktoken

DEFAULT_CHECKPOINT = BASE_DIR / 'checkpoints' / 'last.pt'


# ----------------------------------------------------------
# Simples data loader, avança pelo arquivo em chunks.
# Recebe tokens já separados em treino ou validação. Assim, os dois
# conjuntos não compartilham exemplos e não precisamos tokenizar duas vezes.
# Anotações originais do estudo:
# ----------------------------------------------------------
# Simples data loader, avança pelo arquivo em chunks.
# Bom o suficiente por enquanto.
# Observação sobre a versão atual: O carregamento/tokenização agora acontece em train(); o loader recebe os tokens já separados.
class DataLoaderLite:
    def __init__(self, B, T, tokens):
        if B <= 0 or T <= 0:
            raise ValueError('B e T precisam ser positivos.')
        if len(tokens) < B * T + 1:
            raise ValueError(f'Dados insuficientes: precisamos de {B*T+1} tokens, temos {len(tokens)}.')
        self.B, self.T = B, T
        self.tokens = tokens
        # Anotações originais do estudo:
        # estado
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # Se carregar o próximo batch for out of bounds, reset.
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        buffer = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buffer[:-1].view(B, T) # inputs
        y = buffer[1:].view(B, T)  # targets: cada posição tenta prever o token à direita
        # Anotações originais do estudo:
        # avance a posição do tensor
        self.current_position += B * T
        return x, y


def get_learnrate(step, max_steps, warmup_steps, max_learnrate):
    # Anotações originais do estudo:
    # 1) warmup linear para iteracoes de warmup steps
    # 2) se it > lr_decay_iters, return min learning rate
    # 3) no meio, use o cosseno de decay até o min learning rate.
    # Observação sobre a versão atual: Aqui o decay_ratio é limitado a 1; max_steps define o fim do agendamento.
    # 1) warmup linear: começamos com atualizações menores.
    if step < warmup_steps:
        return max_learnrate * (step + 1) / warmup_steps
    min_learnrate = max_learnrate * 0.1
    # 2) após warmup, cosseno de decay até o mínimo no último passo.
    # max(..., 1) também permite demonstrações de apenas um ou dois passos.
    decay_ratio = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - 1 - warmup_steps)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learnrate + coeff * (max_learnrate - min_learnrate)


@torch.no_grad()
def evaluate(model, loader, batches):
    # Validação mede a previsão em texto que não entra no backward.
    # Preservamos o estado anterior do modelo, inclusive quando chamado fora do treino.
    was_training = model.training
    position = loader.current_position
    model.eval()
    loader.current_position = 0
    try:
        total = 0.0
        for _ in range(batches):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            total += loss.item()
        return total / batches
    finally:
        loader.current_position = position
        model.train(was_training)


def save_checkpoint(path, model, optimizer, step, loader, train_config, data_hash):
    # state_dict contém os pesos, não uma cópia do código Python.
    # Salvamos também AdamW, posição do loader e RNG para continuar o treino.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, config=asdict(model.config),
                   model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                   loader_position=loader.current_position, train_config=train_config,
                   data_hash=data_hash, rng_state=torch.get_rng_state())
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    # Só substituímos o último checkpoint depois de escrever o arquivo completo.
    temporary.replace(path)


def load_checkpoint(path):
    # Carregamos na CPU; weights_only restringe o carregamento a tensores e dados simples.
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('Formato de checkpoint incompatível com este script.')
    return payload


# Anotações originais do estudo:
# ======================================================================
# tentativa de gerar o identico ao que temos no `.ipynb` do HuggingFace
# ======================================================================
# Para usarmos o modelo aleatórito, simplesmente não usamos os pesos
# pre-treinados do HuggingFace, visto que a inicialização dos moudulos,
# como dito antes, já é random por padrão.
#
# ==============================================================
#model = GPT.from_pretrained('gpt2') # Esses são os pesos do HF
# ==============================================================
#
# se deixarmos completamente random, os resultados serão
# setenças aleatórias. lixo total.
# Observação sobre a versão atual: Na geração atual escolhemos pesos treinados ou pretrained. Pesos aleatórios servem como referência inicial.
@torch.inference_mode()
def generate(model, tokens, max_new_tokens, temperature=0.8, top_k=50):
    if not tokens or max_new_tokens < 0 or temperature <= 0 or top_k <= 0:
        raise ValueError('Informe tokens iniciais, temperatura/top-k positivos e quantidade de tokens >= 0.')
    was_training = model.training
    model.eval()
    # Modelo e tokens precisam estar no mesmo dispositivo.
    device = next(model.parameters()).device
    # Anotações originais do estudo:
    # prefix tokens
    #tokens = enc.encode("Hello, I'm a language model,") # apos encodar a string, recebemos uma lista de inteiro; os tokens.
    #
    # x agora eh nosso idx
    # que iremos colocar no forward para gerarmos o
    # nosso 9 token em cada uma das 5 linhas.
    # Observação sobre a versão atual: O exemplo acima usava cinco sequências com oito tokens. Agora geramos uma por vez; T depende do prompt.
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    try:
        # Anotações originais do estudo:
        # generate!
        # agora para a geracao em si.
        # temos nosso x, que agora eh: (B, T) Batch = 5 , Time = 8
        #
        # colocar a `seed` em 42
        #
        # em cada iteracao desse codigo estamos adicionando
        # uma coluna de novos indices em cada um dessas linhas
        # em cada uma dessas linhas, novos indices serao `appended`(adicionados/juntos)
        # a nova sequencia, a partir do que vamos fazendo a amostragem.
        #
        # basicamente, prevemos o proximo elemento e o adicionamos.
        # Observação sobre a versão atual: O exemplo original fixava a seed em 42. Agora --seed controla isso em main(); B=1 nesta função.
        for _ in range(max_new_tokens):
            # O GPT-2 tem uma janela máxima de contexto. Guardamos todo o texto,
            # mas encaminhamos só os últimos block_size tokens em cada passagem.
            context = x[:, -model.config.block_size:]
            # Anotações originais do estudo:
            # encaminhar o modelo para pegar os logits
            # `no_grad` - no gradient, desativa o cálculo de gradientes fazendo com que o
            # pyTorch nao possa executar backpropagation, economizando memoria e processamento.
            # nao precisamos treinar nada, apenas gerar; logo nao precisamos de backpropagation.
            # Observação sobre a versão atual: Aqui inference_mode cobre a função inteira, também desativando o autograd.
            logits, _ = model(context)
            # Anotações originais do estudo:
            # pegar os logits da ultima posicao
            # apenas nos importamos com os logits da ultima coluna.
            logits = logits[:, -1, :] / temperature
            # 50304 inclui padding; tiktoken só conhece IDs de 0 até 50256.
            valid_vocab = min(50257, logits.size(-1))
            logits = logits[:, :valid_vocab]
            # Anotações originais do estudo:
            # pegar os top-k 50 das amostras (padrao do huggingface)
            # ;restrigir a escolha aos 50 tokens mais provaveis.
            # restrigir ao top-50 ajuda o modelo a nao sair do caminho,
            # a se perder tao facilmente do contexto dos tokens.
            #
            # topk_probs = matriz(5, 50) ; topk_indices = matriz(5, 50)
            # Observação sobre a versão atual: Aqui os tensores têm shape (1, k); top-k é configurável, com padrão 50.
            values, indices = torch.topk(logits, min(top_k, valid_vocab), dim=-1)
            # Anotações originais do estudo:
            # pegar as probabilidades
            probs = F.softmax(values, dim=-1)
            # Anotações originais do estudo:
            # selecione um token das top-k probabilidades
            ix = torch.multinomial(probs, 1)
            # Anotações originais do estudo:
            # pegues seus indices correspondentes
            xcol = torch.gather(indices, -1, ix)
            # Anotações originais do estudo:
            # `append`(junte) a sequencia
            x = torch.cat((x, xcol), dim=-1)
            if xcol.item() == 50256: # endoftext: fim da geração
                break
        return x[0].tolist()
    finally:
        model.train(was_training)


def train(args):
    if not args.resume and Path(args.checkpoint).exists():
        raise ValueError('Checkpoint já existe: use --resume ou escolha outro --checkpoint.')
    checkpoint = load_checkpoint(args.checkpoint) if args.resume else None
    # Retomar preserva B, T, acumulação e hiperparâmetros do checkpoint.
    # --steps continua sendo o TOTAL desejado de atualizações (não passos extras).
    settings = checkpoint['train_config'] if checkpoint else dict(
        batch_size=args.batch_size, seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum_steps, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay)
    B, T = settings['batch_size'], settings['seq_len']
    grad_accum_steps = settings['grad_accum_steps']
    if min(B, T, grad_accum_steps, args.steps, args.eval_every, args.eval_batches, args.save_every) <= 0:
        raise ValueError('Tamanhos, passos e intervalos devem ser positivos.')
    if settings['learning_rate'] <= 0 or settings['weight_decay'] < 0 or settings['warmup_steps'] < 0:
        raise ValueError('Learning rate deve ser positivo; warmup e weight decay não podem ser negativos.')
    if settings['warmup_steps'] >= args.steps:
        raise ValueError('warmup-steps deve ser menor que steps; para testes curtos, use --warmup-steps 0.')

    raw = Path(args.data).read_bytes()
    data_hash = hashlib.sha256(raw).hexdigest()
    if checkpoint and checkpoint['data_hash'] != data_hash:
        raise ValueError('O texto mudou desde o checkpoint. Use o mesmo arquivo para retomar.')
    # Anotações originais do estudo:
    # quando inicia, carregue os tokens do disco e coloque-os em memoria
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode_ordinary(raw.decode('utf-8')), dtype=torch.long)
    # Reservamos os últimos 10% para validação, sem embaralhar os trechos.
    split = int(len(tokens) * 0.9)
    # Anotações originais do estudo:
    # Get Data Batch, usar o data loader aqui!
    train_loader = DataLoaderLite(B, T, tokens[:split])
    val_loader = DataLoaderLite(B, T, tokens[split:])
    print(f'Tokens: {split} treino / {len(tokens)-split} validação', flush=True)
    print(f'B={B}, T={T}, acumulação={grad_accum_steps}; {B*T*grad_accum_steps} tokens/atualização', flush=True)

    if checkpoint:
        config = GPTconfigs(**checkpoint['config'])
    else:
        # Mesmos componentes do GPT-2, com dimensões menores para aprender na CPU.
        # --model gpt2 mantém as dimensões originais do GPT-2 de 124M parâmetros.
        profile = dict(n_layer=4, n_head=4, n_embd=128, block_size=128) if args.model == 'small' else {}
        for name in ('n_layer', 'n_head', 'n_embd', 'block_size'):
            if getattr(args, name) is not None:
                profile[name] = getattr(args, name)
        config = GPTconfigs(vocab_size=50304, **profile)
    if min(config.n_layer, config.n_head, config.n_embd, config.block_size) <= 0 or config.n_embd % config.n_head:
        raise ValueError('Dimensões positivas; n_embd precisa ser divisível por n_head.')
    if T > config.block_size:
        raise ValueError('seq-len não pode ultrapassar block_size.')
    # Anotações originais do estudo:
    # Get Logits
    #logits, loss = model(x, y) # Captar os logits e o loss
    # Observação sobre a versão atual: No perfil small reduzimos as dimensões; --model gpt2 seleciona a escala original.
    model = GPT(config).to('cpu')
    print(f'Parâmetros únicos: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    # Anotações originais do estudo:
    # ===================================================================================
    # Para calcular um Starting Point razoavel para um rede neural de raciocinio
    # aleatorio, queremos um loss que é 1 dividido pelo tamanho do vocabulario
    # (ou seja, todos os elementos tem uma probabilidade uniforme, em prol de
    # não favorecer nenhum token na inicialização)
    # Então queremos que a probabilidade de qualquer token arbitrário é mais ou menos
    # 1/vocab_size = 1/50527 = 0.00001989772569
    #
    # Então vamos fazer uma checagem de higienização da perda (loss)
    # por que o cross_entropy loss é basicamente o negativo da probabilidade
    # logaritmica.
    #
    # -ln(1/50257) = 10.8249051197; isso é a loss esperada na inicialização.
    #
    # que é bem perto dos nossos 11.
    # isso significa que nossa probabilidade no começa é bem difusa.
    # e com isso podemos começar a otimizar e dizer para a rede neural
    # quais elementos devem vir e em qual ordem.
    #
    # Nesse ponto podemos fazer um 'loss step backward' (loss.backward = Backpropagation)
    # calcular os gradientes e executar uma otimização.
    # ===================================================================================
    #print(loss) #11.0804 no print que é um tensor de um único elemento.
    # Adam alternativa ao Stochastic Radian Descent Optimizer(SGD) que estava
    # sendo usado antes e acelera a otimização. Mais compelxo de entender.
    # otimizador!
    # -----------------
    # aqui estamos overfittando um único batch
    # começa alto e depois cai com o tempo para um número ínfimo
    #
    # Nos não queremos overfittar um único batch, mas de fato otimizar
    # Então vamos fazer um data loader para otimização.
    #optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
    # Observação sobre a versão atual: Esclarecimentos: 1/vocab_size é a PROBABILIDADE uniforme; a loss é ln(vocab_size). O ID correto é 50257 (50304 com padding). SGD significa Stochastic Gradient Descent. O overfit de um único batch era a etapa anterior do vídeo.
    optimizer = model.configure_optimizers(settings['weight_decay'], settings['learning_rate'], torch.device('cpu'))
    start_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        train_loader.current_position = checkpoint['loader_position']
        torch.set_rng_state(checkpoint['rng_state'])
        print(f'Retomando após {start_step} atualizações completas.', flush=True)
        del checkpoint # os pesos já foram copiados; evita mantê-los duas vezes na RAM
    if start_step >= args.steps:
        raise ValueError('Checkpoint já atingiu steps; informe um total maior para continuar.')
    model.train()
    try:
        for step in range(start_step, args.steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # Anotações originais do estudo:
            #_____________________________________________
            # Como estamos pegando o proximo batch, não é esperado o overfit
            # de nenhuma batch se quer.
            #
            # Aqui o esperado é que nossa Loss diminua, mas nem tanto.
            #
            # Basicamente essa queda vem da deleção de tokens que nunca ocorrem
            # Ou seja, jogar seus valores para -inf.
            #
            # Mas nunca chegaremos a 0 por que a quantidade de iteracoes(50) é baixa.
            #_____________________________________________
            # Observação sobre a versão atual: Batches diferentes não impedem overfit. A loss diminui ajustando scores dos alvos, sem deletar tokens ou obrigatoriamente levar logits a -inf. Cinquenta passos não garantem loss baixa.
            loss_accum = 0.0
            for micro_step in range(grad_accum_steps):
                # Anotações originais do estudo:
                # mover nossos tensores em batches para o device (cpu, gpu ou mps)
                # Observação sobre a versão atual: Nesta versão os dados e o modelo já estão na CPU, então não precisamos de .to(device) aqui.
                x, y = train_loader.next_batch()
                # CPU usa float32. Mixed precision do vídeo é uma otimização de hardware.
                logits, loss = model(x, y)
                # Dividimos antes de somar os gradientes para obter a média do batch efetivo.
                # Anotações originais do estudo:
                # diminuindo o loss para quando somarmos, recuperamos o equiaizador.
                # Observação sobre a versão atual: A divisão faz a soma dos gradientes corresponder à média dos microbatches.
                loss = loss / grad_accum_steps
                # Anotações originais do estudo:
                # .item, vai converter o loss que é um tensor de um único item
                # para um único float. Esse float vai ficar na CPU (ou gpu, ou mps).
                # quando .item é chamado pytorch vai mandar o tensor de volta para cpu (gpu ou mps)
                # em memoria e converter para um float que pode ser printado.
                # Observação sobre a versão atual: Aqui a execução é CPU: .item() extrai o escalar; não há transferência da GPU.
                loss_accum += loss.detach().item()
                loss.backward()
                del logits, loss
            # Calcula a norma global e limita gradientes muito grandes.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Anotações originais do estudo:
            #determina e seta o learning rate para essa iteração
            learnrate = get_learnrate(step, args.steps, settings['warmup_steps'], settings['learning_rate'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = learnrate
            optimizer.step()
            dt = time.perf_counter() - t0 # segundos; multiplicamos por 1000 no print
            tokens_processed = B * T * grad_accum_steps
            print(f'step {step+1:4d}/{args.steps} | loss {loss_accum:.6f} | lr {learnrate:.4e} | '
                  f'norm {norm.item():.4f} | tempo {dt*1000:.2f}ms | tok/sec {tokens_processed/dt:.2f}', flush=True)
            # A loss diminui quando os scores dos próximos tokens corretos melhoram.
            # Isso não deleta tokens do vocabulário e não garante generalização.
            if (step+1) % args.eval_every == 0 or step+1 == args.steps:
                val_loss = evaluate(model, val_loader, args.eval_batches)
                print(f'  validação: loss {val_loss:.6f}', flush=True)
            if (step+1) % args.save_every == 0 or step+1 == args.steps:
                save_checkpoint(args.checkpoint, model, optimizer, step+1, train_loader, settings, data_hash)
                print(f'  checkpoint salvo: {args.checkpoint}', flush=True)
    except KeyboardInterrupt:
        # Evitamos salvar uma atualização incompleta; o último arquivo completo continua válido.
        print('Treino interrompido. Para retomar, use o último checkpoint salvo.', flush=True)
        return
    print('Treino concluído. Use o modo generate para carregar os pesos e gerar texto.', flush=True)


def run_generation(args):
    if args.pretrained:
        # Hugging Face fornece apenas os pesos. O forward e a amostragem são os nossos.
        model = GPT.from_pretrained(args.pretrained).to('cpu')
        print(f'Pesos pretrained: {args.pretrained}', flush=True)
    else:
        checkpoint = load_checkpoint(args.checkpoint or DEFAULT_CHECKPOINT)
        del checkpoint['optimizer'] # inferência não precisa dos momentos do AdamW
        model = GPT(GPTconfigs(**checkpoint['config']))
        model.load_state_dict(checkpoint['model'])
        # A geração não utiliza os estados do AdamW; liberamos a memória do checkpoint.
        print(f"Pesos do treino: {checkpoint['step']} atualizações", flush=True)
        del checkpoint
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode_ordinary(args.prompt) or [enc.eot_token]
    for i in range(args.num_sequences):
        output = generate(model, tokens, args.max_new_tokens, args.temperature, args.top_k)
        # Anotações originais do estudo:
        # print o texto que foi gerado.
        # A diferença de geração para o HuggingFace no 'play.ipynb'
        # se da devido a algum parametro do 'pipeline()' do HF.
        # mas se jogarmos esse mesmo código lá, a mesma frase será gerada.
        # Observação sobre a versão atual: Para comparar com Hugging Face precisamos alinhar pesos, prompt e amostragem. Diferenças numéricas e de ambiente também podem alterar a sequência.
        print(f'{i+1}> {enc.decode(output)}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='GPT-2 próprio: treino e geração na CPU.')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    training = subparsers.add_parser('train', help='Treinar com pesos aleatórios ou retomar checkpoint.')
    training.add_argument('--data', type=Path, default=BASE_DIR / 'train_input.txt')
    training.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    training.add_argument('--resume', action='store_true')
    training.add_argument('--model', choices=('small', 'gpt2'), default='small')
    training.add_argument('--batch-size', type=int, default=2)
    training.add_argument('--seq-len', type=int, default=64)
    training.add_argument('--grad-accum-steps', type=int, default=1)
    training.add_argument('--steps', type=int, default=50)
    training.add_argument('--learning-rate', type=float, default=3e-4)
    training.add_argument('--warmup-steps', type=int, default=5)
    training.add_argument('--weight-decay', type=float, default=0.1)
    training.add_argument('--eval-every', type=int, default=10)
    training.add_argument('--eval-batches', type=int, default=2)
    training.add_argument('--save-every', type=int, default=10)
    for name in ('n-layer', 'n-head', 'n-embd', 'block-size'):
        training.add_argument('--' + name, type=int, default=None)
    generation = subparsers.add_parser('generate', help='Gerar usando a nossa implementação.')
    source = generation.add_mutually_exclusive_group()
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--pretrained', choices=('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'))
    generation.add_argument('--prompt', default='First Citizen:')
    generation.add_argument('--max-new-tokens', type=int, default=40)
    generation.add_argument('--temperature', type=float, default=0.8)
    generation.add_argument('--top-k', type=int, default=50)
    generation.add_argument('--num-sequences', type=int, default=1)
    for command in (training, generation):
        command.add_argument('--threads', type=int, default=2)
        command.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error('threads deve ser positivo.')
    if args.mode == 'generate' and (args.num_sequences <= 0 or args.max_new_tokens < 0 or args.temperature <= 0 or args.top_k <= 0):
        parser.error('Verifique num-sequences, max-new-tokens, temperature e top-k.')
    # Anotações originais do estudo:
    #_______________________________________________________________________________________________
    # Garantindo por onde a pipeline do modelo irá passar
    #
    # cpu - Processador; mais lento mas funciona em todos os dispositivos
    # cuda - Compute Unified Device Architecture. Uso de GPUs NVIDIA, se possuírem arquitetura CUDA.
    # mps - Metal Performance Shaders. Uso de GPUs do MAC nos modelos 'M'.
    # ==============================================================================================
    # Observação sobre a versão atual: As alternativas acima são as do vídeo. Este script escolhe CPU explicitamente.
    torch.set_num_threads(args.threads)
    # Anotações originais do estudo:
    # reprodutabilidade ao setar uma seed.
    torch.manual_seed(args.seed)
    print(f'Dispositivo: cpu | threads: {args.threads}', flush=True)
    try:
        if args.mode == 'train':
            train(args)
        else:
            run_generation(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


# Importar este arquivo só define classes/funções; não dispara um treino.
if __name__ == '__main__':
    main()


# =============================================================================
# TOP-K: SELEÇÃO DOS K TOKENS MAIS PROVÁVEIS
# =============================================================================
#
# Top-k significa "os k elementos com maior valor".
# O k indica quantos candidatos queremos selecionar.
#
# Na geração de texto, selecionamos os k tokens com maior probabilidade
# de aparecer na próxima posição.
#
# -----------------------------------------------------------------------------
# EXEMPLO
# -----------------------------------------------------------------------------
#
# Depois de "Eu gosto de", imagine estas probabilidades:
#
#     Próximo token       Probabilidade
#     ---------------------------------
#     café                    40%
#     música                  30%
#     viajar                  15%
#     chocolate               10%
#     pedra                    5%
#
# Com top-k = 3, apenas "café", "música" e "viajar" participam da escolha.
# Os demais candidatos ficam excluídos desse passo.
#
# -----------------------------------------------------------------------------
# SELEÇÃO E SORTEIO
# -----------------------------------------------------------------------------
#
# Top-k seleciona os candidatos; o sorteio escolhe um deles.
#
# O sorteio respeita os pesos relativos: "café" tem mais chance de ser
# escolhido do que "viajar". Não é uma escolha com chances iguais.
#
# =============================================================================
