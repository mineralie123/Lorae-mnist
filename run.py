"""Explicit entry point: running this file starts the full frozen experiment."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import torch
from config import CONFIG, ROOT
from data import split_indices, training_datasets, test_dataset, make_loader
from models import make_models
from training import (configure_runtime, make_optimizer, train_to,
                      extension_decision, restore_checkpoint)
from evaluation import evaluate_all


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def new_run(kind):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = ROOT / 'results' / f'{kind}_{stamp}_{uuid4().hex[:8]}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'config.json', CONFIG)
    np.savez_compressed(folder / 'dataset_indices.npz', **split_indices())
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    write_json(folder / 'environment.json', {
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'torch_cuda_available': torch.cuda.is_available(), 'torch_cuda_build': torch.version.cuda,
        'packages': dict(sorted(versions.items())), 'run_kind': kind,
    })
    sources = sorted(ROOT.glob('*.py')) + [ROOT / 'README.md']
    write_json(folder / 'source_sha256.json', {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()})
    return folder


def main():
    configure_runtime()
    folder = new_run('experiment')
    write_json(folder / 'status.json', {'status': 'running', 'test_evaluated': False})
    print(f'Results: {folder}', flush=True)
    try:
        train_data, validation_data = training_datasets()
        models = make_models()
        initial_m = models['lorae'].M.weight.detach().numpy().copy()
        np.save(folder / 'initial_M.npy', initial_m)
        optimizers, loaders, histories = {}, {}, {}
        for name, model in models.items():
            model.to(device='cpu', dtype=torch.float32)
            target = folder / name
            target.mkdir()
            optimizers[name] = make_optimizer(model)
            loaders[name] = (make_loader(train_data, True), make_loader(validation_data))
            histories[name] = train_to(model, optimizers[name], *loaders[name], [], 10, target)
        decision = extension_decision(histories)
        write_json(folder / 'extension_decision.json', decision)
        print(f'Frozen extension decision: {decision}', flush=True)
        if decision['extend_both']:
            for name, model in models.items():
                target = folder / name
                histories[name] = restore_checkpoint(target / 'checkpoint.pt', model,
                                                      optimizers[name], *loaders[name])
                histories[name] = train_to(model, optimizers[name], *loaders[name],
                                           histories[name], 30, target)
        # No official test images are evaluated before the duration decision is final.
        metrics = evaluate_all(models, make_loader(test_dataset()), histories, initial_m,
                               folder / 'evaluation')
        write_json(folder / 'status.json', {'status': 'complete', 'test_evaluated': True,
                                           'final_epochs': decision['target_epochs']})
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as error:
        write_json(folder / 'status.json', {'status': 'failed',
                                           'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
