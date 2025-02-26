from kan import KANLayer
#from ID import intrinsic_dimension
import torch
import torch.nn as nn

class KanWrapper(nn.Module):
    def __init__(self, layers, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, 
                 scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, 
                 sb_trainable=True, device='cpu', sparse_init=False):
        """
        Initializes the KAN.
       
        Args:
        -----
            layers: list
                - Includes starting and output layer
            Grid Size:
                - Number of splines KanLayer( grid = grid_size ))
            Polynomial: 
                - Degree of polynomial KanLayer( k = polynomial )
            Other arguments:
                - Passed directly to KANLayer - need to decide defaults.
        """
        super(KanWrapper, self).__init__()
       
        # Define the layers of the network
        self.layers = nn.ModuleList()
        self.device = device

        # Layers
        for i in range(0, len(layers) -1 ):
            self.layers.append(KANLayer(
                in_dim=layers[i],
                out_dim=layers[i+1],
                num=num,
                k=k,
                noise_scale=noise_scale,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                scale_sp=scale_sp,
                base_fun=base_fun,
                grid_eps=grid_eps,
                grid_range=grid_range,
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable,
                device=self.device,
                sparse_init=sparse_init
            ))

    def forward(self, x):
        """
        Forward pass through the Kolmogorov-Arnold Network.
       
        Args:
        -----
            x: torch.Tensor
                Input tensor of shape (batch_size, input_dim).
       
        Returns:
        --------
            torch.Tensor
                Output tensor of shape (batch_size, output_dim).
        """
        # Forward pass through each KANLayer
        output = []
        for layer in self.layers:
            x, _, _, _ = layer(x)  # Assuming layer(x) returns four values
            output.append(x)

        return output

        