import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=1, nonlinearity="relu"):
        super(MLP, self).__init__()
        output_dim=output_ch*resolution*resolution
        self.resolution=resolution
        self.output_ch=output_ch
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)


model_types={
    "MLP":MLP,
}
