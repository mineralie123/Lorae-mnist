"""Fixed split and independently seeded loaders for paired training."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config import CONFIG, ROOT


def split_indices():
    indices = np.random.default_rng(CONFIG['seed']).permutation(60000)
    return {'train': indices[:5000], 'validation': indices[5000:6000],
            'unused': indices[6000:], 'test': np.arange(10000, dtype=np.int64)}


def image_transform():
    return transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True), transforms.ToTensor(),
    ])


def training_datasets():
    dataset = datasets.MNIST(ROOT / 'data', train=True, download=True,
                             transform=image_transform())
    assert len(dataset) == 60000
    indices = split_indices()
    return (Subset(dataset, indices['train'].tolist()),
            Subset(dataset, indices['validation'].tolist()))


def test_dataset():
    dataset = datasets.MNIST(ROOT / 'data', train=False, download=True,
                             transform=image_transform())
    assert len(dataset) == CONFIG['test_size']
    return dataset


def make_loader(dataset, training=False):
    generator = torch.Generator(device='cpu').manual_seed(CONFIG['seed'])
    return DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=training,
                      num_workers=0, drop_last=False, pin_memory=False,
                      generator=generator)
