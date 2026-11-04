"""Registra treinos do zero, tempos e amostras, usando o script principal sem alterá-lo."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / 'gpt2mini-default.py'
PATTERN = re.compile(r'step\s+(\d+)/(\d+) \| loss ([\d.e+-]+) \| lr ([\d.e+-]+) \| norm ([\d.e+-]+) \| tempo ([\d.e+-]+)ms \| tok/sec ([\d.e+-]+)')


def worker(checkpoint, output, threads):
    # Processo separado: libera toda a memória do modelo antes do próximo treino.
    import importlib.util
    import platform
    import torch
    import psutil
    spec = importlib.util.spec_from_file_location('gpt_experimento', SCRIPT)
    gpt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gpt
    spec.loader.exec_module(gpt)
    torch.set_num_threads(threads)
    torch.manual_seed(1337)
    device = gpt.select_device()
    payload = None if checkpoint == 'initial' else gpt.load_checkpoint(checkpoint)
    config = gpt.GPTconfigs(n_layer=4, n_head=4, n_embd=128, block_size=128, vocab_size=50304)
    if payload:
        config = gpt.GPTconfigs(**payload['config'])
        del payload['optimizer']
    model = gpt.GPT(config).to(device)
    if payload:
        model.load_state_dict(payload['model'])
        del payload
    enc = gpt.tiktoken.get_encoding('gpt2')
    raw = (BASE / 'train_input.txt').read_bytes()
    tokens = torch.tensor(enc.encode_ordinary(raw.decode('utf-8')), dtype=torch.long)
    split = int(len(tokens) * 0.9)
    # Todos são avaliados no MESMO trecho, com B=1 e T=32. Assim não
    # exigimos posições de 32 a 63 de modelos que só treinaram com T=32.
    train_loader = gpt.DataLoaderLite(1, 32, tokens[:split])
    val_loader = gpt.DataLoaderLite(1, 32, tokens[split:])
    train_loss = gpt.evaluate(model, train_loader, 32)
    val_loss = gpt.evaluate(model, val_loader, 32)
    samples = []
    original_block_size = model.config.block_size
    model.config.block_size = 32 # janela comum: só posições treinadas por todos
    for prompt in ['First Citizen:', 'To be, or not to be', 'Eu estou aprendendo']:
        torch.manual_seed(42)
        start = time.perf_counter()
        ids = enc.encode_ordinary(prompt)
        generated = gpt.generate(model, ids, 40, temperature=0.8, top_k=50)
        gpt.synchronize(device)
        duration = time.perf_counter() - start
        samples.append(dict(prompt=prompt, seed=42, temperature=0.8, top_k=50,
                            text=enc.decode(generated), new_tokens=len(generated)-len(ids),
                            seconds=duration, context_window=32))
    model.config.block_size = original_block_size
    result = dict(device=str(device), torch_version=torch.__version__, python=sys.version,
                  platform=platform.platform(), processor=platform.processor(), threads=threads,
                  ram_total_bytes=psutil.virtual_memory().total,
                  ram_available_bytes=psutil.virtual_memory().available,
                  parameters=sum(p.numel() for p in model.parameters()), config=gpt.asdict(config),
                  data_sha256=hashlib.sha256(raw).hexdigest(), total_tokens=len(tokens),
                  train_tokens=split, val_tokens=len(tokens)-split,
                  evaluation=dict(batch_size=1, seq_len=32, batches=32, tokens=1024),
                  train_loss_common=train_loss, val_loss_common=val_loss,
                  samples=samples, weights_origin='random_initialization' if checkpoint=='initial' else 'own_training')
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE/'experimentos'/'2026-11-04')
    parser.add_argument('--budget-tokens', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--worker')
    parser.add_argument('--worker-output')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.worker_output, args.threads)
        return
    root = args.output.resolve()
    if args.budget_tokens <= 0 or args.threads <= 0:
        parser.error('Orçamento e threads devem ser positivos.')
    if root.exists() and any(root.glob('exp*/checkpoints/last.pt')):
        raise FileExistsError('A pasta já contém treinos. Escolha outro --output para preservar o registro.')
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    baseline = root/'inicial.json'
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', 'initial',
                    '--worker-output', str(baseline), '--threads', str(args.threads)], check=True, env=env)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
                    budget_tokens=args.budget_tokens, seed=1337, runs=[])
    for name, B, T in [('exp01_B1_T32',1,32), ('exp02_B2_T32',2,32), ('exp03_B2_T64',2,64)]:
        if args.budget_tokens % (B*T):
            raise ValueError('O orçamento precisa ser divisível por todos os valores B*T.')
        steps = args.budget_tokens // (B*T)
        folder = root/name
        folder.mkdir(exist_ok=True)
        checkpoint = folder/'checkpoints'/'last.pt'
        if checkpoint.exists():
            raise FileExistsError(f'Experimento já existe: {checkpoint}. Use outro --output.')
        command = [sys.executable,str(SCRIPT),'train','--model','small','--batch-size',str(B),
                   '--seq-len',str(T),'--steps',str(steps),'--warmup-steps','5',
                   '--learning-rate','3e-4','--threads',str(args.threads),'--seed','1337',
                   '--grad-accum-steps','1','--eval-every',str(max(1,steps//8)),
                   '--eval-batches',str(512//(B*T)),'--save-every',str(steps),'--checkpoint',str(checkpoint)]
        (folder/'comando.json').write_text(json.dumps(command,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'Iniciando {name}: {steps} atualizações, {args.budget_tokens} tokens',flush=True)
        start = time.perf_counter()
        process = subprocess.Popen(command,cwd=BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        rows = []
        with (folder/'treino.log').open('w',encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line); log.flush()
                match = PATTERN.search(line)
                if match:
                    step,_,loss,lr,norm,ms,tps = match.groups()
                    rows.append(dict(step=int(step),tokens_seen=int(step)*B*T,train_loss=float(loss),
                                     learning_rate=float(lr),gradient_norm=float(norm),
                                     step_ms=float(ms),tokens_per_second=float(tps),val_loss=''))
                    if int(step)%max(1,steps//4)==0:
                        print(line.strip(),flush=True)
                elif 'validação: loss ' in line and rows:
                    rows[-1]['val_loss']=float(line.rsplit(' ',1)[-1])
        if process.wait():
            raise RuntimeError(f'Treino falhou: consulte {folder / "treino.log"}')
        wall = time.perf_counter()-start
        with (folder/'metricas.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        result_path = folder/'resultado.json'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(checkpoint),
                        '--worker-output',str(result_path),'--threads',str(args.threads)],check=True,env=env)
        metadata['runs'].append(dict(name=name,B=B,T=T,steps=steps,tokens=args.budget_tokens,
                                     training_wall_seconds=wall,training_step_seconds=sum(r['step_ms'] for r in rows)/1000,
                                     result=str(result_path.relative_to(root)),metrics=str((folder/'metricas.csv').relative_to(root))))
        (root/'registro.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata['finished_utc']=datetime.now(timezone.utc).isoformat()
    (root/'registro.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Experimentos concluídos:',root,flush=True)


if __name__=='__main__':
    main()
