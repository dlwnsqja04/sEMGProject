"""DenseNet161 전체 학습셋 5에폭 학습과 실제 loss 기록."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet161

from week2_signal import OUT


def train(epochs: int = 5) -> Path:
    """매 에폭 3800개 전체를 사용하며 테스트 데이터는 학습에 넣지 않는다."""
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    prepared = json.loads((OUT / 'preparation.json').read_text())
    x = np.load(OUT / 'dataset/X_train.npy', allow_pickle=False)
    y = np.load(OUT / 'dataset/y_train.npy', allow_pickle=False)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    model = densenet161(weights=None)
    model.classifier = nn.Linear(2208, 5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = OUT / 'densenet161_checkpoint.pt'
    start_epoch = 0
    history: list[tuple[int, float, float, float]] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['fingerprint'] != prepared['fingerprint']:
            message = '데이터셋이 변경되었습니다. 기존 학습 결과를 별도 보관한 뒤 재학습하세요.'
            raise RuntimeError(message)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f'Saved checkpoint: {start_epoch} completed epochs', flush=True)
    config = {'model': 'DenseNet161', 'weights': None, 'classes': 5, 'batch_size': 16,
              'optimizer': 'Adam', 'learning_rate': 0.001, 'criterion': 'CrossEntropyLoss',
              'seed': 42, 'device': str(device), 'torch_version': str(torch.__version__),
              'dataset_fingerprint': prepared['fingerprint'], 'train_samples': len(dataset),
              'loss_aggregation': 'sample-weighted mean over all training windows',
              'test_used_for_training': False, 'test_evaluation': 'not performed in week 2'}
    if not (OUT / 'training_config.json').exists() or start_epoch < epochs:
        (OUT / 'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log_path = OUT / 'training_log.txt'
    with log_path.open('a' if start_epoch else 'w', encoding='utf-8') as log:
        if start_epoch == 0:
            header = (f'DenseNet161 | weights=None | classes=5 | device={device}\n'
                      'Adam lr=0.001 | batch_size=16 | seed=42\n'
                      f'Train: {len(dataset)} windows from 200 trials | Test held out: 950 windows\n')
            log.write(header)
            log.flush()
            print(header, end='', flush=True)
        for epoch in range(start_epoch + 1, epochs + 1):
            generator = torch.Generator().manual_seed(42 + epoch)
            loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=generator, num_workers=0)
            model.train()
            total_loss, correct, seen = 0., 0, 0
            started = time.monotonic()
            for step, (xb, yb) in enumerate(loader, 1):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f'유한하지 않은 loss: epoch={epoch}, batch={step}')
                loss.backward()
                optimizer.step()
                total_loss += loss_value * len(yb)
                correct += int((logits.argmax(dim=1) == yb).sum().item())
                seen += len(yb)
                if step % 50 == 0:
                    print(f'Epoch {epoch}/{epochs} | batch {step}/{len(loader)} | loss_so_far={total_loss / seen:.6f}', flush=True)
            elapsed = time.monotonic() - started
            epoch_loss, accuracy = total_loss / seen, correct / seen
            history.append((epoch, epoch_loss, accuracy, elapsed))
            line = f'Epoch {epoch:02d}/{epochs:02d} | loss={epoch_loss:.6f} | train_acc={accuracy:.4f} | samples={seen} | seconds={elapsed:.1f}'
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
            with (OUT / 'training_history.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['epoch', 'loss', 'train_accuracy', 'seconds'])
                writer.writerows(history)
            temporary = checkpoint_path.with_suffix('.partial')
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'history': history, 'fingerprint': prepared['fingerprint']}, temporary)
            os.replace(temporary, checkpoint_path)
    print(f'Completed epochs: {len(history)}; log: {log_path}', flush=True)
    return log_path


if __name__ == '__main__':
    train()
