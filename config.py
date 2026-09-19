"""Frozen CISC3024 protocol. Command-line overrides are intentionally absent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = {
    'seed': 42, 'device': 'cpu', 'dtype': 'float32', 'cpu_threads': 4,
    'train_size': 5000, 'validation_size': 1000, 'test_size': 10000,
    'image_size': 32, 'latent_dim': 128, 'batch_size': 32, 'num_workers': 0,
    'initial_epochs': 10, 'extended_epochs': 30, 'extension_threshold': 0.01,
    'optimizer': 'Adam', 'learning_rate': 0.001, 'betas': [0.9, 0.999],
    'epsilon': 1e-8, 'weight_decay': 0.0, 'lambda': 0.001,
    'reconstruction_reduction': 'mean_over_examples_and_pixels',
    'resize_interpolation': 'bilinear', 'antialias': True,
    'augmentation': False, 'normalization': None, 'drop_last': False,
    'shuffle_train': True, 'shuffle_evaluation': False,
    'split_algorithm': 'numpy.random.default_rng(42).permutation(60000)',
    'checkpoint_selection': 'final_epoch_after_extension_decision',
    'interpolation_pairs': [[0, 1], [3, 8], [4, 9]], 'interpolation_steps': 11,
    'architecture': {
        'encoder_channels': [1, 32, 64, 128, 256],
        'encoder_convolution': {'kernel': 4, 'stride': 2, 'padding': 1},
        'encoder_fc': [1024, 128], 'decoder_fc': [128, 8192],
        'decoder_reshape': [128, 8, 8],
        'decoder_transpose_convolutions': [[128, 64, 4, 2, 1],
                                            [64, 32, 4, 2, 1],
                                            [32, 1, 1, 1, 0]],
        'hidden_activation': 'ReLU', 'output_activation': 'Tanh',
        'M': {'shape': [128, 128], 'bias': False, 'activation': None},
    },
}
