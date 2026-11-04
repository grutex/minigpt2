from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import subprocess
import time
import inspect

# Versao com thread padrao.
#============================================================================
# Se possuir uma GPU NVIDIA com CUDA, instale o torch na versao CUDA.
# o codigo ira verificar se eh possivel usar CUDA. Se nao, ira usar a CPU.
#
# OBS:
# Esse codigo estava crashando meu PC, provavelmente super aquecimento da CPU.
#
#
# Info:
# 6 threads (padrao PyTorch)
# 30 tokens (padrao HuggingFace)
# 5 sequencias (padrao HuggingFace)
#============================================================================

#Ugly x Beautiful Numbers
# Esses compiladores rodam em potencias de 2
# Ugly Numbers são aqueles que não são divisiveis por potencia de 2
# Beautiful Numbers: 128, 768, 1024, 2048 . . .
# Ugly: 50257
# Sempre tente remover os ugly numbers, quem sabe
# talvez você consiga um pouco de performance por isso...
#
# Para resolver isso, aumente o numero ao número que é potencia de 2
# mais proximo; ex: 50257 -> 50304, divisivel por 8, 16, 32, 64 . . .

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
        # hoje tanh é mais por saudosismo. o GELU pode ser usado diretamente.
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
        # aqui tem ~30% dos parametros sendo poupados ao usar esse Scheme.
        # 768 * 50257 = 38,597,376 parametros ~40M
        # 40M/124M = 0.3225 (32%)
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
#-=-=-=-=-=-=-  FIM GPT-2  -=-=-=-=-=-=-=-=--=-=


#=============  GERAÇÃO  =======================

# ----------------------------------------------------------
# Simples data loader, avança pelo arquivo em chunks.
# Bom o suficiente por enquanto.
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # quando inicia, carregue os tokens do disco e coloque-os em memoria
        with open('train_input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"Carregado {len(self.tokens)} tokens.")
        print(f"1 epoca = {len(self.tokens) // (B * T)} batches")

        # estado
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buffer[:-1].view(B, T)) #inputs
        y = (buffer[1:].view(B, T)) #targets
        
        # avance a posição do tensor
        self.current_position += B * T
        
        # se carregar o proximo batch for out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y
# ----------------------------------------------------------

#
#
# Única função desse código é limpar o terminal para ficar
# trazer uma visualização mais limpa.
subprocess.run(
    'cls' if os.name == 'nt' else 'clear',
    shell=os.name == 'nt',
    check=True,
)
#_______________________________________________________________________________________________
# Garantindo por onde a pipeline do modelo irá passar
#
# cpu - Processador; mais lento mas funciona em todos os dispositivos
# cuda - Compute Unified Device Architecture. Uso de GPUs NVIDIA, se possuírem arquitetura CUDA.
# mps - Metal Performance Shaders. Uso de GPUs do MAC nos modelos 'M'.
# ==============================================================================================
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"Dispositivo: {device}")

# reprodutabilidade ao setar uma seed.
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 4
T = 128
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"Total batch size requerida: {total_batch_size}")
print(f"=> steps calculados do gradiente acumulado: {grad_accum_steps}")

# Get Data Batch, usar o data loader aqui!
train_loader = DataLoaderLite(B=B, T=T)

# Get Logits
model = GPT(GPTconfigs(vocab_size=50304)) # Criar o modelo, padrão mini 124M params.
model.to(device)
#logits, loss = model(x, y) # Captar os logits e o loss

max_learnrate = 6e-4
min_learnrate = max_learnrate * 0.1
warmup_steps = 10
max_steps = 50
def get_learnrate(it):
    # 1) warmup linear para iteracoes de warmup steps
    if it < warmup_steps:
        return max_learnrate * (it + 1) / warmup_steps
    # 2) se it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_learnrate
    # 3) no meio, use o cosseno de decay até o min learning rate.
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff começa com 1 e vai para 0
    return min_learnrate + coeff * (max_learnrate - min_learnrate)


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
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
   
    optimizer.zero_grad() # começar com um gradiente zerado.
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
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # mover nossos tensores em batches para o device (cpu, gpu ou mps)
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)

        # diminuindo o loss para quando somarmos, recuperamos o equiaizador.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() #detahc é detach o tensor para manter apenas os valores
        loss.backward() # quando pegar o loss e rolar o backpropag, adiciona ao gradiente
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # calcula o normal global de todos os parametros.

    #determina e seta o learning rate para essa iteração
    learnrate = get_learnrate(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learnrate 
    
    optimizer.step() # Atualizar o sparametros e diminuir o loss
    # .item, vai converter o loss que é um tensor de um único item
    # para um único float. Esse float vai ficar na CPU (ou gpu, ou mps).
    # quando .item é chamado pytorch vai mandar o tensor de volta para cpu (gpu ou mps)
    # em memoria e converter para um float que pode ser printado.
    
    t1 = time.time()
    dt = t1-t0 # diferença em milisegundos.
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tok_p_sec = tokens_processed / dt

    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {learnrate:.4e} | norm: {norm:.4f} | downtime: {dt*1000:.2f}ms | tok/sec: {tok_p_sec:.2f}") 


import sys; sys.exit(0) # Serve para pular a lógica anterior abaixo



# ======================================================================
# tentativa de gerar o identico ao que temos no `.ipynb` do HuggingFace
num_return_sequences = 5 # numero de gerações
max_length = 30 # max length de tokens
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
model = GPT(GPTconfigs()) # o padrão é o mini (124M params).
print("\n\t|| nao crashou, gerando sentenças... ||")
print("="*100 + "\n")
model.eval() # coloca o modelo em evaluation mode: boa pratica para nao treinar e apenas usar.
model.to(device) # move os parametros do modelo para o dispositivo setado em 'device' (cpu, cuda ou mps).

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
#tokens = enc.encode("Hello, I'm a language model,") # apos encodar a string, recebemos uma lista de inteiro; os tokens.
tokens = enc.encode("Peter Piper picked a peck of pickled peppers?")
tokens = torch.tensor(tokens, dtype=torch.long) # (, 8) tokens
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # entao vamos replicar os 8 tokens, 5 vezes (5, 8) tokens
#
# x agora eh nosso idx
# que iremos colocar no forward para gerarmos o
# nosso 9 token em cada uma das 5 linhas.
x = tokens.to(device) # tokens e modelo precisam estar no mesmo dispositivo.

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
torch.manual_seed(42) # as respostas sao geradas a partir dessa seed, sempre serão as mesmas.
if device.type == 'cuda':
    torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # encaminhar o modelo para pegar os logits
    # `no_grad` - no gradient, desativa o cálculo de gradientes fazendo com que o 
    # pyTorch nao possa executar backpropagation, economizando memoria e processamento.
    # nao precisamos treinar nada, apenas gerar; logo nao precisamos de backpropagation.
    with torch.no_grad(): 
        logits = model(x) # (B, T, vocab_size)
        # pegar os logits da ultima posicao
        # apenas nos importamos com os logits da ultima coluna.
        logits = logits[:, -1, :] # (B, vocab_size)
        # pegar as probabilidades
        probs = F.softmax(logits, dim=-1)

        # pegar os top-k 50 das amostras (padrao do huggingface)
        # ;restrigir a escolha aos 50 tokens mais provaveis.
        # restrigir ao top-50 ajuda o modelo a nao sair do caminho,
        # a se perder tao facilmente do contexto dos tokens.
        #
        # topk_probs = matriz(5, 50) ; topk_indices = matriz(5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        # selecione um token das top-k probabilidades
        ix = torch.multinomial(topk_probs, 1)

        # pegues seus indices correspondentes
        xcol = torch.gather(topk_indices, -1, ix)

        # `append`(junte) a sequencia
        x = torch.cat((x, xcol), dim=-1) 

# print o texto que foi gerado.
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"{i+1}> {decoded} \n")
# A diferença de geração para o HuggingFace no 'play.ipynb'
# se da devido a algum parametro do 'pipeline()' do HF.
# mas se jogarmos esse mesmo código lá, a mesma frase será gerada.
print("="*100)


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
