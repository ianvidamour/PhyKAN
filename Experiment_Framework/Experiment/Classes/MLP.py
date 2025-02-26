import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, dimensions, device):
        super(CustomMLP, self).__init__()
        self.device = device
        layers = []
        num_layers = len(dimensions) - 1
        for i in range(num_layers):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < num_layers - 1:  # Add ReLU activation for all but the last layer
                layers.append(nn.ReLU())
        
        
        # Use nn.Sequential to stack the layers
        self.model = nn.Sequential(*layers).to(device)

        self.model[-1].bias=torch.nn.Parameter(torch.tensor([-2.2136, -2.0723, -2.2079, -2.1656, -2.2308, -2.3044, -2.2008, -2.1446, -2.2276, -2.2183], device=self.device))

    def forward(self, x):
        layer_outputs = []
        x = x.to(self.device)
        
        # Iterate through each layer, but only save outputs from nn.Linear layers
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                layer_outputs.append(x)
        
        return layer_outputs  # Return the list containing outputs of each Linear layer
