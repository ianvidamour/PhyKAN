from kan import KANLayer
#from ID import intrinsic_dimension
import torch
import torch.nn as nn

class KanWrapper(nn.Module):
    def __init__(self, layers, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, 
                 scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, 
                 sb_trainable=True, device='cpu', sparse_init=False, l2Loss=False, batch=False):
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
            l2Loss:
                - Informs on whether to include custom loss function
            Other arguments:
                - Passed directly to KANLayer - need to decide defaults.
        """
        super(KanWrapper, self).__init__()
       
        # Define the layers of the network
        self.layers = nn.ModuleList()
        self.device = device

        self.l2Loss = l2Loss

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

            if batch == True:    
                layers.append(nn.BatchNorm1d(layers[i +1]))
                layers.append(nn.ReLU())
        
        self.to(device)

        self.postacts = []

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
        if not (self.l2Loss):
            for layer in self.layers:
                x, _, _, _ = layer(x)  # Assuming layer(x) returns four values
                output.append(x)

        else:
            self.postacts = []
            for layer in self.layers:
                x, _, _, postacts = layer(x)  # Assuming layer(x) returns four values
                output.append(x)
                self.postacts.append(postacts)

        return output

    def loss_fn(self):

        # postacts each have dimension of (batch, prev, next), so self.postacts[0].shape[0] is the batch size
        entropy = torch.zeros( size = ((self.postacts[0].shape[0]), ), dtype=torch.float32)
        
        for postact in self.postacts:
            
            abs_pa = abs( postact)

            # Dim = (1,2) sums accross previous to next layers activation(i,j)
            phi = torch.sum( abs_pa , dim = (1,2) ) # phi.shape is [batch_size] for this one layer
            #phi = phi.view(phi.shape[0], 1, 1)  

            # Divide each postact in batch by phi, the view is so that it works accross postact dimension
            layer_entropy = abs_pa / phi.view(phi.shape[0], 1, 1) # torch.Size([batch_size, prev, next])

            # Complete entropy calculation
            layer_entropy = layer_entropy.sum(dim=(1,2)) # torch.Size([batch_size])

            layer_entropy = layer_entropy * torch.log(layer_entropy)
            
            # Add layer entropy to total
            entropy  += layer_entropy + phi

        return entropy



'''
            entropy = 

            phi = [ for ] # Use postacts for the penalty 
            
            
            # Check that postacts is 3D tensor with size (batch, previous, nextlayer)
            # (100, 784, 200)

        penalty = 0
        for activation in self.postacts:
            penatly += abs(activation) / layer

        output.append(x)'''