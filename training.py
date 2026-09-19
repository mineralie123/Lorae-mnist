"""CPU training, checkpoint state, and the frozen extension decision."""
import csv
import json
import random
import time
import numpy as np
import torch
from config import CONFIG
from models import loss_terms


def configure_runtime():
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(CONFIG['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                            betas=tuple(CONFIG['betas']), eps=CONFIG['epsilon'],
                            weight_decay=0.0)


@torch.inference_mode()
def reconstruction_mse(model, loader, max_batches=None):
    model.eval()
    squared_error, pixels = 0.0, 0
    for batch_index, (x, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device='cpu', dtype=torch.float32)
        error = (model(x) - x).square().sum().item()
        if not np.isfinite(error):
            raise FloatingPointError('Nonfinite evaluation loss; do not extend training.')
        squared_error += error
        pixels += x.numel()
    if not pixels:
        raise ValueError('Empty evaluation loader.')
    return squared_error / pixels


def train_epoch(model, optimizer, loader, max_batches=None):
    model.train()
    totals = {'train_reconstruction_mse': 0.0, 'train_total_objective': 0.0,
              'train_nuclear_norm': 0.0, 'train_weighted_penalty': 0.0}
    count, regularized = 0, False
    for batch_index, (x, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device='cpu', dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        rec, nuclear, penalty, objective = loss_terms(model, x)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite objective; stop and investigate.')
        objective.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite gradient; stop and investigate.')
        optimizer.step()
        size = x.size(0)
        count += size
        totals['train_reconstruction_mse'] += rec.item() * size
        totals['train_total_objective'] += objective.item() * size
        if nuclear is not None:
            regularized = True
            totals['train_nuclear_norm'] += nuclear.item() * size
            totals['train_weighted_penalty'] += penalty.item() * size
    if not count:
        raise ValueError('Empty training loader.')
    result = {key: value / count for key, value in totals.items()}
    if not regularized:
        result['train_nuclear_norm'] = None
        result['train_weighted_penalty'] = None
    return result


def save_checkpoint(path, model, optimizer, history, train_loader, validation_loader):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'history': history, 'config': CONFIG,
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(),
             'train_generator': train_loader.generator.get_state(),
             'validation_generator': validation_loader.generator.get_state()}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, train_loader, validation_loader):
    # Load only this run's own trusted local checkpoint; it includes Python RNG state.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config'] != CONFIG:
        raise ValueError('Checkpoint does not match the frozen protocol.')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    train_loader.generator.set_state(state['train_generator'])
    validation_loader.generator.set_state(state['validation_generator'])
    return state['history']


def write_history(path, history):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_to(model, optimizer, train_loader, validation_loader, history, target, folder):
    cumulative = history[-1]['cumulative_training_seconds'] if history else 0.0
    for epoch in range(len(history) + 1, target + 1):
        start = time.perf_counter()
        row = train_epoch(model, optimizer, train_loader)
        row['validation_reconstruction_mse'] = reconstruction_mse(model, validation_loader)
        elapsed = time.perf_counter() - start
        cumulative += elapsed
        row = {'epoch': epoch, **row, 'epoch_seconds': elapsed,
               'cumulative_training_seconds': cumulative}
        history.append(row)
        write_history(folder / 'history.csv', history)
        save_checkpoint(folder / 'checkpoint.pt', model, optimizer, history,
                        train_loader, validation_loader)
        print(f'{folder.name} epoch {epoch}/{target}: '
              f'train MSE={row["train_reconstruction_mse"]:.6f}, '
              f'validation MSE={row["validation_reconstruction_mse"]:.6f}, '
              f'{elapsed:.1f}s', flush=True)
    return history


def extension_decision(histories):
    detail = {}
    for name in ('ae', 'lorae'):
        history = histories[name]
        if [row['epoch'] for row in history] != list(range(1, 11)):
            raise ValueError('Extension decision requires exactly epochs 1 through 10.')
        values = np.array([row['validation_reconstruction_mse'] for row in history])
        if not np.isfinite(values).all():
            raise FloatingPointError('Invalid validation history.')
        a, b = float(values[4:7].mean()), float(values[7:10].mean())
        improvement = (a - b) / max(a, 1e-12)
        detail[name] = {'mean_epochs_5_7': a, 'mean_epochs_8_10': b,
                        'relative_improvement': improvement,
                        'triggers_extension': improvement >= CONFIG['extension_threshold']}
    extend = any(value['triggers_extension'] for value in detail.values())
    return {'extend_both': extend, 'target_epochs': 30 if extend else 10,
            'models': detail}
