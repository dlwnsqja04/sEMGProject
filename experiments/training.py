"""학습·검증 기록과 검증 기준 체크포인트 선택."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet161, mobilenet_v3_small, resnet18

from experiments.data import OUT, load_split


def create_model(name: str) -> nn.Module:
    if name == 'resnet18':
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif name == 'densenet161':
        model = densenet161(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    elif name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
    else:
        raise ValueError(f'지원하지 않는 모델: {name}')
    return model


def score_predictions(probabilities: np.ndarray, y: np.ndarray, ids: np.ndarray) -> dict:
    trial_y, trial_prediction = [], []
    for trial in np.unique(ids):
        mask = ids == trial
        if len(np.unique(y[mask])) != 1:
            raise ValueError('한 시행에 서로 다른 라벨이 있습니다.')
        trial_y.append(int(y[mask][0]))
        trial_prediction.append(int(probabilities[mask].mean(axis=0).argmax()))
    prediction = probabilities.argmax(axis=1)
    return {'window_accuracy': accuracy_score(y, prediction),
            'window_f1': f1_score(y, prediction, average='macro', labels=range(5), zero_division=0),
            'trial_accuracy': accuracy_score(trial_y, trial_prediction),
            'trial_f1': f1_score(trial_y, trial_prediction, average='macro', labels=range(5), zero_division=0),
            'trial_confusion': confusion_matrix(trial_y, trial_prediction, labels=range(5)).tolist(),
            'window_confusion': confusion_matrix(y, prediction, labels=range(5)).tolist()}


def rank(metrics: dict) -> tuple:
    return metrics['trial_f1'], metrics['window_f1'], -metrics['val_loss']


def evaluate(model: nn.Module, data: tuple, device: torch.device) -> tuple[dict, np.ndarray]:
    x, y, ids = data
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=32)
    model.eval()
    total, probabilities = 0., []
    with torch.inference_mode():
        for xb, yb in loader:
            logits = model(xb.to(device))
            total += float(nn.functional.cross_entropy(logits, yb.to(device), reduction='sum').item())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    p = np.concatenate(probabilities)
    metrics = score_predictions(p, y, ids)
    metrics['val_loss'] = total / len(y)
    return metrics, p


def train(name: str, width: int, hop: int, folder: Path, stage: str, epochs: int,
          device: torch.device, patience: int = 100) -> dict:
    destination = OUT / 'runs' / f'{stage}_{name}_w{width}_h{hop}'
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination / 'complete.json'
    if complete.exists():
        prior = json.loads(complete.read_text())
        if prior['requested_epochs'] >= epochs:
            if prior['completed_epochs'] > epochs:
                with (destination / 'history.csv').open(newline='') as stream:
                    earlier = [
                        {key: int(value) if key == 'epoch' else float(value)
                         for key, value in row.items()}
                        for row in csv.DictReader(stream) if int(row['epoch']) <= epochs
                    ]
                if len(earlier) != epochs:
                    raise RuntimeError(f'{destination.name}: {epochs} epochs의 기록이 없습니다.')
                return {**prior, 'requested_epochs': epochs, 'completed_epochs': epochs,
                        'best': max(earlier, key=rank)}
            return prior
    torch.manual_seed(42)
    np.random.seed(42)
    model = create_model(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-5)
    x, y, _ = load_split(folder, 'train')
    validation = load_split(folder, 'val')
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    history, best, stale = [], None, 0
    last = destination / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, stale = saved['history'], saved['best'], saved['stale']
        torch.set_rng_state(saved['rng'])
    print(f'RUN {destination.name}: epochs {len(history)} -> {epochs}', flush=True)
    for epoch in range(len(history) + 1, epochs + 1):
        started = time.monotonic()
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(42 + epoch))
        model.train()
        total_loss, correct = 0., 0
        learning_rate = optimizer.param_groups[0]['lr']
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError(f'{destination.name}, epoch {epoch}: loss={value}')
            loss.backward()
            optimizer.step()
            total_loss += value * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
        metrics, _ = evaluate(model, validation, device)
        row = {'epoch': epoch, 'train_loss': total_loss / len(y), 'train_accuracy': correct / len(y),
               **{k: v for k, v in metrics.items() if 'confusion' not in k},
               'lr': learning_rate, 'seconds': time.monotonic() - started}
        history.append(row)
        scheduler.step(metrics['val_loss'])
        if best is None or rank(row) > rank(best):
            best, stale = row.copy(), 0
            torch.save({'model': model.state_dict(), 'best': best, 'name': name,
                        'width': width, 'hop': hop}, destination / 'best.pt')
        else:
            stale += 1
        with (destination / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        line = (f'{destination.name} epoch={epoch:02d} loss={row["train_loss"]:.6f} '
                f'val_loss={row["val_loss"]:.6f} val_trial_F1={row["trial_f1"]:.4f} '
                f'val_window_F1={row["window_f1"]:.4f} seconds={row["seconds"]:.1f}')
        with (destination / 'training_log.txt').open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
        if epoch >= 10 and stale >= patience:
            break
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'history': history, 'best': best,
                'stale': stale, 'rng': torch.get_rng_state()}, last)
    result = {'name': name, 'width': width, 'hop': hop, 'stage': stage,
              'requested_epochs': epochs, 'completed_epochs': len(history), 'best': best,
              'parameters': sum(p.numel() for p in model.parameters()), 'directory': str(destination)}
    complete.write_text(json.dumps(result, indent=2))
    del model, optimizer, dataset, x, validation
    if device.type == 'mps':
        torch.mps.empty_cache()
    return result
