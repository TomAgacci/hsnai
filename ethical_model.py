import torch
import torch.nn as nn
import torch.optim as optim
import math

# Golden ratio band
phi = (1 + math.sqrt(5)) / 2.0
L = 1.0 / phi
U = phi
x_star = 1.0

alpha = 2.0      # sharpness of golden-mean fitness
eta = 1e-3       # main learning rate
lambda_ = 1e-3   # corruption/repair correction rate
beta = 1.0       # HNS truth weight

class EthicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = MyModel()  # your base model

    def forward(self, x):
        return self.core(x)

def intensity_metric(outputs):
    # Example: "ethical intensity" or extremity
    return outputs.abs().mean()

def force_configuration(outputs):
    # Nietzschean "force" proxy: gradient magnitude or variance
    return outputs.pow(2).mean()

def necessity_fixed_point(force_values, steps=5):
    # crude iterative contraction toward a "fixed point"
    v = force_values
    for _ in range(steps):
        v = 0.5 * (v + v.detach().mean())
    return v

def heidegger_unconcealment(necessity):
    # Heideggerian clearing: map necessity to a truth-weight in [0,1]
    return torch.sigmoid(beta * (necessity - necessity.detach().mean()))
