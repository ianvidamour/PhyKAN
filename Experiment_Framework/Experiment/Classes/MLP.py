import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, dimensions, device, batch=False):
        super(CustomMLP, self).__init__()
        self.device = device
        layers = []
        num_layers = len(dimensions) - 1

        for i in range(num_layers):
            in_dim = dimensions[i]
            out_dim = dimensions[i + 1]

            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))

            # Add BN + ReLU for all layers except the last
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())

        # Use nn.Sequential to stack the layers
        self.model = nn.Sequential(*layers).to(device)


    def forward(self, x):
        layer_outputs = []
        x = x.to(self.device)

        # Iterate through each layer and save outputs from Linear layers only
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                layer_outputs.append(x)

        return layer_outputs  # Return outputs of each Linear layer