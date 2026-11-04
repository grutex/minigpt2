# Experimentos de treinamento próprio

Os três modelos foram inicializados aleatoriamente, sem pesos GPT-2 pretrained.
O tokenizer GPT-2 é reutilizado, e a rede foi treinada com train_input.txt.

Cada treino processou 4.096 posições, na CPU, com duas threads, seed 1337,
quatro blocos, quatro cabeças e embeddings de 128 dimensões (7.248.640 parâmetros).
A janela máxima da arquitetura é 128 e o vocabulário tem 50.304 saídas.

| Experimento | B | T | Atualizações | Tokens processados |
| --- | --- | --- | --- | --- |
| exp01_B1_T32 | 1 | 32 | 128 | 4096 |
| exp02_B2_T32 | 2 | 32 | 64 | 4096 |
| exp03_B2_T64 | 2 | 64 | 32 | 4096 |

As perdas finais de resumo.csv foram reavaliadas com B=1, T=32 e 32 batches,
sempre nos mesmos 1.024 tokens iniciais de cada conjunto. Os demais valores
periódicos de validação em metricas.csv usam o contexto do próprio treino.
Nas amostras finais também usamos janela comum de 32 posições: isso evita usar
posições não treinadas pelos modelos cujo T era 32. O limite é de 40 tokens novos,
com seed 42 reiniciada para cada prompt, top-k=50 e temperatura 0.8.

O orçamento é igual em tokens, mas o número de atualizações e a proporção de warmup
mudam. A comparação não isola o efeito de batch/contexto sobre a aprendizagem.
Cada configuração foi medida uma vez, sequencialmente, sem controlar a carga dos
outros programas nem a temperatura. Tempos e vazões não devem ser generalizados.

## Arquivos

- registro.json: comandos de referência, hashes e tempos das execuções.
- inicial.json: avaliação e amostras da inicialização aleatória.
- resumo.csv e resumo.json: métricas finais comparáveis.
- AMOSTRAS.md: todas as saídas, sem corrigir o texto gerado.
- curvas_loss.png: figura usada no relatório.
- testes.log: execução dos nove testes, todos aprovados.
- Cada pasta exp*: comando.json, treino.log, metricas.csv, resultado.json e
  checkpoints/last.pt. Os checkpoints ficam locais e são ignorados pelo Git.

Tokens/s no resumo é o total de tokens dividido pela soma dos tempos dos passos.
Tempo total inclui inicialização, tokenização, validação e salvamento; não inclui
a reavaliação posterior nem a geração das amostras em outro processo.

## Reproduzir

Na raiz do projeto, com as dependências e o ambiente virtual preparados:

```powershell
python registrar_experimentos.py --output experimentos/nova_execucao
```

Para repetir a avaliação e as amostras de um checkpoint existente:

```powershell
python registrar_experimentos.py --worker experimentos/2026-11-04/exp01_B1_T32/checkpoints/last.pt --worker-output tmp/resultado_repetido.json
```

Para gerar livremente com os pesos treinados, usando o limite de contexto original:

```powershell
python gpt2mini-default.py generate --checkpoint experimentos/2026-11-04/exp01_B1_T32/checkpoints/last.pt --prompt "First Citizen:" --seed 42
```

O comando livre pode produzir texto diferente do registro experimental, pois usa
a janela máxima original de 128 em vez da janela comum de 32 usada na comparação.
Os dados são um pequeno trecho do corpus e não treinam o GPT-2 completo de 124M.
