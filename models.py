"""Official MNIST architecture with an explicit, separate LoRAE transformation."""
import torch
from torch import nn
from torch.nn import functional as F
from config import CONFIG


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in zip([1, 32, 64, 128], [32, 64, 128, 256]):
            layers.extend([nn.Conv2d(cin, cout, 4, 2, 1), nn.ReLU()])
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(1024, 128)

    def forward(self, x):
        return self.fc(self.convs(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 8192)
        self.convs = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 1, 1, 0), nn.Tanh(),
        )

    def forward(self, z):
        return self.convs(self.fc(z).reshape(-1, 128, 8, 8))


class StandardAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


class LoRAE(StandardAE):
    def __init__(self):
        super().__init__()
        # Only this weight matrix receives nuclear-norm regularization.
        self.M = nn.Linear(128, 128, bias=False)

    def encode(self, x):
        return self.M(self.encoder(x))


def make_models():
    torch.manual_seed(CONFIG['seed'])
    ae = StandardAE()
    lorae = LoRAE()
    # Copy values, not parameter objects: optimizers must remain independent.
    lorae.encoder.load_state_dict(ae.encoder.state_dict())
    lorae.decoder.load_state_dict(ae.decoder.state_dict())
    return {'ae': ae, 'lorae': lorae}


def loss_terms(model, x):
    reconstruction = F.mse_loss(model(x), x, reduction='mean')
    if isinstance(model, LoRAE):
        nuclear = torch.linalg.matrix_norm(model.M.weight, ord='nuc')
        penalty = CONFIG['lambda'] * nuclear
        return reconstruction, nuclear, penalty, reconstruction + penalty
    # The baseline has neither M nor a regularizer, even in the computation graph.
    return reconstruction, None, None, reconstruction
