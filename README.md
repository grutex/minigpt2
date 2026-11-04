# GPT-2 próprio: treino e geração na CPU

## Passo a passo: treinar seus próprios pesos e gerar texto

O modo `train` inicia com pesos aleatórios e aprende com `train_input.txt`.
O modo `generate` carrega os pesos salvos desse treinamento. Pesos pretrained
só são usados quando a opção `--pretrained` é informada explicitamente.

Na pasta do projeto, com o ambiente virtual ativado:

```powershell
# 1. Treinar do zero na CPU e salvar em checkpoints/last.pt.
python gpt2mini-default.py train

# 2. Gerar texto usando os pesos que você mesmo treinou.
python gpt2mini-default.py generate --prompt "First Citizen:" --max-new-tokens 40

# 3. Continuar o treinamento até completar 200 atualizações no total.
python gpt2mini-default.py train --resume --steps 200
```

Se já existir um checkpoint, retome com `--resume` ou escolha outro arquivo
para começar um experimento do zero:

```powershell
python gpt2mini-default.py train --checkpoint checkpoints/experimento2.pt
python gpt2mini-default.py generate --checkpoint checkpoints/experimento2.pt --prompt "First Citizen:"
```

O padrão treina uma versão menor da arquitetura GPT-2 para reduzir o custo na CPU.
Você está treinando seus próprios pesos, mas não o GPT-2 completo de 124M.
Para essa escala, use `--model gpt2`, com custo muito maior de memória e tempo.
As 50 atualizações padrão demonstram o processo; texto coerente exige mais
treinamento e depende dos dados, do tamanho do modelo e dos hiperparâmetros.

## Preparar o ambiente

Execute `mkinit_venv.cmd` para criar/ativar o ambiente e instalar as dependências.
Os comandos abaixo usam o Python desse ambiente ativado. Para ver todas as opções:

```powershell
python gpt2mini-default.py --help
python gpt2mini-default.py train --help
python gpt2mini-default.py generate --help
```

## Treinar na CPU

```powershell
python gpt2mini-default.py train
```

O padrão usa B=2, T=64, duas threads, uma acumulação e 50 atualizações.
Para reduzir o trabalho por atualização, use `--batch-size 1 --seq-len 32`.
O perfil `small` usa quatro blocos, quatro cabeças e embeddings de 128 dimensões:
os componentes são os mesmos da arquitetura GPT-2, com escala menor para esta CPU.
Isso NÃO equivale ao modelo original de 124M nem aos seus pesos pretrained.
Um treino curto iniciado com pesos aleatórios provavelmente gera texto sem coerência.

O arquivo `train_input.txt` é tokenizado com o vocabulário GPT-2. Os primeiros 90%
dos tokens vão para treino e os últimos 10% para validação. A loss de validação é
impressa a cada dez atualizações e no final. `tok/sec` mede posições de tokens
processadas durante o treino; não é a velocidade de geração.

```powershell
# Mesmo tamanho do GPT-2 de 124M, com vocabulário preenchido até 50304.
# Exige muito mais RAM e tempo; não é o padrão recomendado para esta máquina.
python gpt2mini-default.py train --model gpt2 --batch-size 1 --seq-len 32 --checkpoint checkpoints/gpt2.pt

# Exemplo de controle da acumulação: dois microbatches por atualização.
python gpt2mini-default.py train --grad-accum-steps 2 --checkpoint checkpoints/accum.pt
```

`--threads` limita o paralelismo; `--learning-rate`, `--warmup-steps` e
`--weight-decay` controlam a otimização. `--n-layer`, `--n-head`, `--n-embd` e
`--block-size` permitem explorar outras dimensões. n_embd deve ser divisível por
n_head, e seq-len não pode ultrapassar block-size. A execução usa float32 em CPU.

## Salvar e retomar

O treino salva `checkpoints/last.pt` a cada dez atualizações e no final. Ajuste isso
com `--save-every`. Escolha outro arquivo com `--checkpoint` para iniciar um novo
experimento; o script não sobrescreve um treino anterior sem `--resume`.

```powershell
python gpt2mini-default.py train --resume --steps 100
```

`--steps 100` significa atingir 100 atualizações no total. Se o checkpoint contém
50, serão executadas mais 50. O script restaura a arquitetura, o AdamW, o estado
aleatório, a posição no texto e os hiperparâmetros de treino. As opções de modelo,
B, T, learning rate e acumulação de uma nova execução não substituem as salvas.
O total de steps informado define o horizonte do decay; aumentá-lo estende a curva
de aprendizado. Para reproduzir uma execução interrompida, mantenha seu total original.
O texto deve ser o mesmo: seu hash é conferido antes de retomar.

Ctrl+C interrompe o processo. O último checkpoint completo permanece utilizável;
as atualizações posteriores ao último salvamento serão refeitas ao retomar.

## Gerar com os pesos treinados

```powershell
python gpt2mini-default.py generate --prompt "First Citizen:" --max-new-tokens 40
python gpt2mini-default.py generate --checkpoint checkpoints/gpt2.pt --prompt "Once upon a time"
```

A geração carrega a configuração e os pesos do checkpoint e usa nosso próprio
forward, softmax, top-k e sorteio. Não cria outro modelo com pesos aleatórios para
substituí-los. A temperatura padrão é 0.8, top-k=50 e uma sequência por execução.
Use `--temperature`, `--top-k`, `--num-sequences` e `--seed` para experimentar.
O texto completo é preservado, mas o forward recebe só a janela de contexto mais
recente. Os IDs extras de padding do vocabulário são excluídos da amostragem.

## Gerar com pesos GPT-2 pretrained

```powershell
python gpt2mini-default.py generate --pretrained gpt2 --prompt "Once upon a time"
```

Essa opção baixa os pesos quando necessário e os copia para nossas classes.
O processamento do modelo e a geração continuam na nossa implementação, sem
pipeline pronta. O GPT-2 pretrained é um completador de texto, não um chatbot.
Seu uso deve ser identificado no relatório: esses pesos não foram treinados aqui.
O checkpoint local de treino não é necessário para esse comando.

## Verificações pequenas

```powershell
python -m unittest -v test_gpt2mini
```

Os testes verificam causalidade, gradientes, compartilhamento de pesos, acumulação,
limites da geração e o fluxo real de treinar, salvar, retomar e gerar. Usam modelos
pequenos e arquivos temporários; não baixam pesos pretrained nem treinam o GPT-2 completo.
O tokenizer GPT-2 fica no cache local `.cache/tiktoken` (ou TIKTOKEN_CACHE_DIR,
se essa variável já estiver configurada). O primeiro uso
pode precisar de internet, inclusive para iniciar o treino.

`crash-test.py` permanece como o experimento anterior de geração com pesos GPT-2.
O script principal contém o fluxo completo e os checkpoints usados pelos comandos acima.
