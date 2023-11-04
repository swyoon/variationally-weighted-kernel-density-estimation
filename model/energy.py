import torch
import torch.nn as nn

class Score_network(nn.Module):
    def __init__(
        self,
        input_dim,
        units,
        SiLU=True,
        dropout=True
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.SiLU() if SiLU else nn.ReLU(),
                nn.Dropout(.7) if dropout else nn.Identity()
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class Weight_network(nn.Module):
    def __init__(
            self,
            input_dim,
            units,
            SiLU=True,
            dropout=True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units[:-1]:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.SiLU() if SiLU else nn.ReLU(),
                nn.Dropout(.7) if dropout else nn.Identity()
            ])
            in_dim = out_dim

        layers.extend([
                nn.Linear(in_dim, units[-1]),
                nn.Sigmoid(),
                nn.Dropout(.5) if dropout else nn.Identity(),
                nn.Linear(units[-1], 1)
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Energy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.net(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True)[0]
    
    def minus_forward(self, x):
        return - self.net(x)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self