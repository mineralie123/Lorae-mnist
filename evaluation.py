"""Final held-out evaluation and plots. No classification or generative fitting."""
import os
import json
import numpy as np
import torch
from config import CONFIG, ROOT
os.environ['MPLCONFIGDIR'] = str(ROOT / '.venv' / 'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finish_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_learning(histories, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in histories.items():
        epochs = [r['epoch'] for r in rows]
        line, = ax.plot(epochs, [r['train_reconstruction_mse'] for r in rows],
                        label=f'{name.upper()} train')
        ax.plot(epochs, [r['validation_reconstruction_mse'] for r in rows], '--',
                color=line.get_color(), label=f'{name.upper()} validation')
    ax.set(xlabel='Epoch', ylabel='Mean squared reconstruction error')
    ax.legend()
    finish_figure(fig, output / 'reconstruction_curves.png')
    rows = histories['lorae']
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [('train_reconstruction_mse', 'Reconstruction MSE'),
                       ('train_weighted_penalty', 'Weighted nuclear penalty'),
                       ('train_total_objective', 'Total objective')]:
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], label=label)
    ax.set(xlabel='Epoch', ylabel='Example-weighted training average')
    ax.legend()
    finish_figure(fig, output / 'lorae_objective_curves.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['epoch'] for r in rows], [r['train_nuclear_norm'] for r in rows])
    ax.set(xlabel='Epoch', ylabel='Mean training nuclear norm of M')
    finish_figure(fig, output / 'lorae_nuclear_norm_curve.png')


@torch.inference_mode()
def collect_evaluation(model, loader):
    model.eval()
    embeddings, selected, squared_error, pixels = [], {d: [] for d in range(10)}, 0.0, 0
    for x, labels in loader:
        x = x.to('cpu', dtype=torch.float32)
        z = model.encode(x)
        reconstruction = model.decoder(z)
        error = (reconstruction - x).square().sum().item()
        if not np.isfinite(error) or not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite final evaluation.')
        squared_error += error
        pixels += x.numel()
        embeddings.append(z.numpy())
        for image, label in zip(x, labels):
            digit = int(label)
            if len(selected[digit]) < 2:
                selected[digit].append(image.clone())
    if any(len(images) != 2 for images in selected.values()):
        raise ValueError('Evaluation requires at least two examples of every digit.')
    z = np.concatenate(embeddings).astype(np.float64)
    covariance = np.cov(z, rowvar=False, ddof=1)
    spectrum = np.linalg.svd(covariance, compute_uv=False)
    total = spectrum.sum()
    if total > 0:
        cumulative = np.cumsum(spectrum) / total
        dimensions = {str(p): int(np.searchsorted(cumulative, p / 100) + 1)
                      for p in (95, 99)}
        normalized = spectrum / spectrum[0]
    else:
        cumulative = np.zeros_like(spectrum)
        normalized = np.zeros_like(spectrum)
        dimensions = {'95': 0, '99': 0}
    images = torch.stack([selected[d][i] for i in range(2) for d in range(10)])
    return {'mse': squared_error / pixels, 'count': len(z), 'images': images,
            'covariance': covariance, 'spectrum': spectrum, 'normalized': normalized,
            'cumulative': cumulative, 'dimensions': dimensions}


@torch.inference_mode()
def evaluate_all(models, loader, histories, initial_m, output, split_label='test'):
    output.mkdir(exist_ok=False)
    plot_learning(histories, output)
    collected, metrics = {}, {}
    for name, model in models.items():
        values = collect_evaluation(model, loader)
        collected[name] = values
        np.savez_compressed(output / f'{name}_latent_analysis.npz',
                            covariance=values['covariance'],
                            singular_values=values['spectrum'],
                            normalized_singular_values=values['normalized'],
                            cumulative_variance=values['cumulative'])
        metrics[name] = {
            f'{split_label}_reconstruction_mse': values['mse'],
            'evaluation_examples': values['count'],
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'cumulative_training_seconds': histories[name][-1]['cumulative_training_seconds'],
            'final_epoch': histories[name][-1]['epoch'],
            'effective_dimension_95': values['dimensions']['95'],
            'effective_dimension_99': values['dimensions']['99'],
        }
    assert torch.equal(collected['ae']['images'], collected['lorae']['images'])
    with (output / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump({'evaluation_split': split_label, 'models': metrics}, handle, indent=2,
                  allow_nan=False)
    import csv
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', *metrics['ae']])
        writer.writeheader()
        for name, row in metrics.items():
            writer.writerow({'model': name, **row})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, values in collected.items():
        axes[0].semilogy(np.arange(1, 129), np.maximum(values['normalized'], 1e-16),
                         label=name.upper())
        axes[1].plot(np.arange(1, 129), values['cumulative'], label=name.upper())
    axes[0].set(xlabel='Singular-value index', ylabel='Value / largest value',
                title='Latent covariance spectrum', ylim=(1e-16, 1))
    axes[1].set(xlabel='Number of components', ylabel='Cumulative variance', ylim=(0, 1.01))
    for level in (0.95, 0.99):
        axes[1].axhline(level, linestyle=':', color='gray')
    for ax in axes:
        ax.legend()
    finish_figure(fig, output / 'latent_covariance_spectra.png')

    final_m = models['lorae'].M.weight.detach().numpy().copy()
    initial_s = np.linalg.svd(initial_m, compute_uv=False)
    final_s = np.linalg.svd(final_m, compute_uv=False)
    np.savez_compressed(output / 'M_analysis.npz', initial_M=initial_m, final_M=final_m,
                        initial_singular_values=initial_s, final_singular_values=final_s)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, 129), initial_s, label='Initialization')
    ax.plot(np.arange(1, 129), final_s, label='Final')
    ax.set(xlabel='Singular-value index', ylabel='Singular value of M')
    ax.legend()
    finish_figure(fig, output / 'M_singular_values.png')

    images = collected['ae']['images']
    reconstructions = {name: model(images) for name, model in models.items()}
    np.savez_compressed(output / 'reconstruction_examples.npz', originals=images.numpy(),
                        ae=reconstructions['ae'].numpy(), lorae=reconstructions['lorae'].numpy())
    fig, axes = plt.subplots(6, 10, figsize=(12, 8))
    for example in range(2):
        for row, (label, batch) in enumerate([('Original', images),
                                             ('AE', reconstructions['ae']),
                                             ('LoRAE', reconstructions['lorae'])]):
            for digit in range(10):
                ax = axes[example * 3 + row, digit]
                ax.imshow(batch[example * 10 + digit, 0].clamp(0, 1),
                          cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if digit == 0:
                    ax.set_ylabel(label)
                if example == row == 0:
                    ax.set_title(str(digit))
    finish_figure(fig, output / 'matched_reconstructions.png')

    fig, axes = plt.subplots(6, 11, figsize=(14, 8))
    interpolation_arrays = {}
    for pair_index, (left, right) in enumerate(CONFIG['interpolation_pairs']):
        for model_index, (name, model) in enumerate(models.items()):
            endpoints = model.encode(images[[left, right]])
            alpha = torch.linspace(0, 1, 11).unsqueeze(1)
            decoded = model.decoder((1 - alpha) * endpoints[0] + alpha * endpoints[1])
            interpolation_arrays[f'{name}_{left}_{right}'] = decoded.numpy()
            for step in range(11):
                ax = axes[2 * pair_index + model_index, step]
                ax.imshow(decoded[step, 0].clamp(0, 1), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if step == 0:
                    ax.set_ylabel(f'{name.upper()}\n{left} to {right}')
                if pair_index == model_index == 0:
                    ax.set_title(f'{step / 10:.1f}')
    finish_figure(fig, output / 'fixed_interpolations.png')
    np.savez_compressed(output / 'interpolation_examples.npz', **interpolation_arrays)
    return metrics
