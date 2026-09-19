"""Limited checks only. Official test images are never evaluated here."""
import copy
import json
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from config import CONFIG
from data import split_indices, training_datasets, make_loader
from models import make_models, loss_terms
from training import (configure_runtime, make_optimizer, train_epoch, reconstruction_mse,
                      extension_decision, save_checkpoint, restore_checkpoint, write_history)
from evaluation import evaluate_all
from run import new_run, write_json


def decision_fixture(early, late):
    return [{'epoch': e, 'validation_reconstruction_mse':
             early if e <= 7 else late} for e in range(1, 11)]


def main():
    configure_runtime()
    folder = new_run('smoke')
    print(f'Smoke artifacts: {folder}', flush=True)
    write_json(folder / 'status.json', {'status': 'running', 'official_test_evaluated': False})
    checks = []
    try:
        indices = split_indices()
        expected = np.random.default_rng(42).permutation(60000)
        assert np.array_equal(indices['train'], expected[:5000])
        assert np.array_equal(indices['validation'], expected[5000:6000])
        assert np.array_equal(indices['test'], np.arange(10000))
        assert len(np.unique(np.concatenate([indices['train'], indices['validation'],
                                            indices['unused']]))) == 60000
        checks.append('Exact fixed permutation, sizes, and disjoint split')

        index_data = TensorDataset(torch.arange(5000), torch.arange(5000))
        order_a = torch.cat([x for x, _ in make_loader(index_data, True)])
        order_b = torch.cat([x for x, _ in make_loader(index_data, True)])
        assert torch.equal(order_a, order_b)
        assert len(torch.unique(order_a)) == 5000
        assert len(list(make_loader(index_data, True))[-1][0]) == 8
        checks.append('Identical seeded ordering; no dropped final partial batch')

        models = make_models()
        ae, lorae = models['ae'], models['lorae']
        for a, b in zip(ae.parameters(), list(lorae.encoder.parameters()) +
                        list(lorae.decoder.parameters())):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        assert not hasattr(ae, 'M')
        assert lorae.M.weight.shape == (128, 128) and lorae.M.bias is None
        assert all(p.device.type == 'cpu' and p.dtype == torch.float32
                   for m in models.values() for p in m.parameters())
        for model in models.values():
            convs = [m for m in model.encoder.convs if isinstance(m, nn.Conv2d)]
            assert [(m.in_channels, m.out_channels) for m in convs] == [(1,32),(32,64),(64,128),(128,256)]
            assert all(m.kernel_size == (4,4) and m.stride == (2,2) and
                       m.padding == (1,1) and m.bias is not None for m in convs)
            assert (model.encoder.fc.in_features, model.encoder.fc.out_features) == (1024,128)
            assert (model.decoder.fc.in_features, model.decoder.fc.out_features) == (128,8192)
            deconvs = [m for m in model.decoder.convs if isinstance(m, nn.ConvTranspose2d)]
            assert [(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding)
                    for m in deconvs] == [(128,64,(4,4),(2,2),(1,1)),
                                        (64,32,(4,4),(2,2),(1,1)),
                                        (32,1,(1,1),(1,1),(0,0))]
            assert isinstance(model.decoder.convs[-1], nn.Tanh)
        checks.append('Frozen layer specifications, identical unshared E/D initialization, CPU float32')
        initial_m = lorae.M.weight.detach().numpy().copy()
        np.save(folder / 'initial_M.npy', initial_m)

        print('Loading MNIST training split for two-batch smoke checks only...', flush=True)
        train_data, validation_data = training_datasets()
        x, _ = next(iter(make_loader(train_data)))
        assert x.shape == (32,1,32,32) and x.dtype == torch.float32
        assert x.min() >= 0 and x.max() <= 1
        for model in models.values():
            assert model.encoder(x[:4]).shape == (4,128)
            assert model.encode(x[:4]).shape == (4,128)
            assert model(x[:4]).shape == (4,1,32,32)
        rec, nuc, penalty, total = loss_terms(ae, x[:4])
        assert nuc is None and penalty is None and total is rec
        rec, nuc, penalty, total = loss_terms(lorae, x[:4])
        assert nuc.item() > 0 and penalty.item() > 0
        assert torch.allclose(total, rec + penalty)
        g_rec = torch.autograd.grad(rec, lorae.M.weight, retain_graph=True)[0]
        g_penalty = torch.autograd.grad(penalty, lorae.M.weight, retain_graph=True)[0]
        g_total = torch.autograd.grad(total, lorae.M.weight)[0]
        assert torch.isfinite(g_rec).all() and g_rec.norm() > 0
        assert torch.isfinite(g_penalty).all() and g_penalty.norm() > 0
        assert torch.allclose(g_total, g_rec + g_penalty, atol=1e-7, rtol=1e-4)
        gradient_report = {'reconstruction_gradient_norm': g_rec.norm().item(),
                           'penalty_gradient_norm': g_penalty.norm().item(),
                           'total_gradient_norm': g_total.norm().item(),
                           'nuclear_norm': nuc.item(), 'weighted_penalty': penalty.item()}
        checks.append('Tensor shapes/range; AE has no penalty; reconstruction and penalty both differentiate through M')

        histories, optimizers, loaders = {}, {}, {}
        for name, model in models.items():
            target = folder / name
            target.mkdir()
            optimizer = make_optimizer(model)
            optimizers[name] = optimizer
            loaders[name] = (make_loader(train_data, True), make_loader(validation_data))
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            start = time.perf_counter()
            row = train_epoch(model, optimizer, loaders[name][0], max_batches=2)
            row['validation_reconstruction_mse'] = reconstruction_mse(
                model, loaders[name][1], max_batches=1)
            elapsed = time.perf_counter() - start
            row = {'epoch': 1, **row, 'epoch_seconds': elapsed,
                   'cumulative_training_seconds': elapsed}
            histories[name] = [row]
            assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
            if name == 'lorae':
                assert not torch.equal(before['M.weight'], model.M.weight)
            write_history(target / 'history.csv', [row])
            save_checkpoint(target / 'checkpoint.pt', model, optimizer, [row], *loaders[name])
            print(f'{name}: two training batches + one validation batch passed ({elapsed:.2f}s)', flush=True)
        checks.append('Two real MNIST batches per model; finite losses; parameters and M update')

        flat = decision_fixture(1.0, 1.0)
        improvement = decision_fixture(1.0, 0.98)
        boundary = decision_fixture(1.0, 0.99)
        below = decision_fixture(1.0, 0.9901)
        worse = decision_fixture(1.0, 1.1)
        assert extension_decision({'ae':flat,'lorae':flat})['target_epochs'] == 10
        assert extension_decision({'ae':improvement,'lorae':flat})['target_epochs'] == 30
        assert extension_decision({'ae':flat,'lorae':improvement})['target_epochs'] == 30
        assert extension_decision({'ae':boundary,'lorae':flat})['target_epochs'] == 30
        assert extension_decision({'ae':below,'lorae':worse})['target_epochs'] == 10
        bad = copy.deepcopy(flat); bad[5]['validation_reconstruction_mse'] = float('nan')
        try:
            extension_decision({'ae':bad,'lorae':flat})
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Nonfinite decision input not rejected')
        checks.append('Extension rule: no/either/boundary/below/worsening/nonfinite cases')

        # Resume two copies from the same saved state and compare one further update.
        continuation_states, rng_draws, next_batches = [], [], []
        continuation_data = TensorDataset(x[:4], torch.zeros(4, dtype=torch.long))
        for _ in range(2):
            clone = copy.deepcopy(lorae)
            opt = make_optimizer(clone)
            train_loader, val_loader = make_loader(train_data, True), make_loader(validation_data)
            restored = restore_checkpoint(folder / 'lorae' / 'checkpoint.pt', clone, opt,
                                          train_loader, val_loader)
            assert restored == histories['lorae'] and len(opt.state) > 0
            rng_draws.append((random.random(), float(np.random.random()), torch.rand(3)))
            next_batches.append(next(iter(train_loader))[0])
            train_epoch(clone, opt, make_loader(continuation_data), max_batches=1)
            continuation_states.append({k:v.clone() for k,v in clone.state_dict().items()})
        assert rng_draws[0][:2] == rng_draws[1][:2]
        assert torch.equal(rng_draws[0][2], rng_draws[1][2])
        assert torch.equal(next_batches[0], next_batches[1])
        assert all(torch.equal(v, continuation_states[1][k]) for k,v in continuation_states[0].items())
        checks.append('Checkpoint restores optimizer/RNG/loader state; continuation update is identical')

        # Synthetic fixtures exercise every final plot without touching official test data.
        synthetic_images = torch.rand(20,1,32,32, generator=torch.Generator().manual_seed(42))
        synthetic_labels = torch.arange(10).repeat(2)
        fixture = TensorDataset(synthetic_images, synthetic_labels)
        metrics = evaluate_all(models, make_loader(fixture), histories, initial_m,
                               folder / 'synthetic_evaluation', split_label='smoke_synthetic')
        assert all(m['evaluation_examples'] == 20 for m in metrics.values())
        expected_pngs = ['reconstruction_curves.png','lorae_objective_curves.png',
                         'lorae_nuclear_norm_curve.png','latent_covariance_spectra.png',
                         'M_singular_values.png','matched_reconstructions.png','fixed_interpolations.png']
        for filename in expected_pngs:
            assert (folder / 'synthetic_evaluation' / filename).stat().st_size > 0
        checks.append('All seven plots and numerical outputs generated on synthetic fixtures only')
        report = {'status':'passed', 'official_test_evaluated':False,
                  'full_training_started':False, 'checks':checks, 'M_gradient_check':gradient_report,
                  'real_train_batches_per_model':2, 'real_validation_batches_per_model':1,
                  'synthetic_evaluation_examples':20,
                  'parameter_counts':{name:sum(p.numel() for p in m.parameters())
                                      for name,m in models.items()},
                  'smoke_losses':histories}
        write_json(folder / 'smoke_report.json', report)
        write_json(folder / 'status.json', {'status':'passed', 'official_test_evaluated':False})
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        write_json(folder / 'status.json', {'status':'failed', 'official_test_evaluated':False,
                                           'error':f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
