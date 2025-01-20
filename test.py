import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class PhyLAN(nn.Module):
    def __init__(self, LANshape, filters_per_unit):
        super().__init__()
        self.Nlayers = len(LANshape)-1
        self.LANshape = LANshape
        # Create parameters for each layer
        self.filter_params = []
        self.weights = []
        for i in range(self.Nlayers):
            layer_params = torch.sqrt(torch.tensor(6/(LANshape[i]+LANshape[i+1])))*(torch.rand((LANshape[i], LANshape[i+1]))-0.5)
            layer_params.requires_grad=True
            self.weights.append(layer_params)
            node_params = 4*(torch.rand((LANshape[i+1], filters_per_unit, 3))-0.5)
            self.filter_params.append(node_params)
        self.node_optim = torch.optim.Adam(params=self.filter_params, lr=1e-3)
        self.edge_optim = torch.optim.Adam(params=self.weights, lr=1e-3)
        self.lossfn = nn.MSELoss()

    def band_pass(self, Xin, params):
        # Bound parameters
        params=params.sigmoid()
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        freq = 10**(3.69+2*Xin).unsqueeze(-1)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 10*(params[:,:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10+(10**(0.5+5*params[:,:, 1].unsqueeze(0)))
        fc_high = 10+(10**(0.5+5*params[:,:, 2].unsqueeze(0)))
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        return Hout.sum(dim=-1)
    
    # Redefined for torch & assuming trainable parameters
    def band_pass_single(self, Xin, params):
        # Bound parameters
        params=params.sigmoid()
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 10*(params[:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10+(10**(0.5+5*params[:, 1].unsqueeze(0)))
        fc_high = 10+(10**(0.5+5*params[:, 2].unsqueeze(0)))
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        return Hout.sum(dim=1).sum(dim=-1)

    def forward(self, Xin):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.LANshape[i+1])))
            outputs.append(torch.zeros((batch_size, self.LANshape[i+1])))
        inputs[0][:, :] = torch.matmul(Xin, self.weights[0])
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.band_pass(inputs[layer], self.filter_params[layer])      
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = torch.matmul(outputs[layer][:, :], self.weights[layer+1])
        return outputs[-1]
        
    def train(self, Xin, Yin):
        # Reset optimisers
        self.node_optim.zero_grad()
        self.edge_optim.zero_grad()
        # Pass through Model
        pred = self.forward(Xin)
        # Calculate loss
        loss = self.lossfn(pred, Yin)
        # Backward pass
        loss.backward()
        # Update parameters
        self.node_optim.step()
        self.edge_optim.step()
        return(loss.item())


# Now for a 2D function instead:
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)

model = PhyLAN([2, 200, 50, 1], 6)
from tqdm import tqdm
for i in tqdm(range(100000)):
    Xs = torch.rand(1000, 2)
    Ys = f(2*(Xs-0.5))
    loss = model.train(Xs, Ys)
    if i%1000 == 0:
        print(loss)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        
        xs = torch.rand(10000, 2)
        y_true = f(2*(xs-0.5))
        y_pred = model.forward(xs)
        
        
        xs = xs.cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
    
        ax.scatter(xs[:, 0], xs[:, 1], y_true, label='True Function', alpha=0.5, s=5)
        ax.scatter(xs[:, 0], xs[:, 1], y_pred, label='Modelled Function', alpha=0.5, s=5)
        plt.legend()
        plt.show()

#%%
xin  = torch.linspace(0, 1, 1000)
filter_params = model.filter_params
print(len(filter_params))

for l, layer in enumerate(filter_params):
    for node in range(10):
        params = layer[node]
        print(params)
        yf = model.band_pass_single(xin, params)
        plt.figure()
        plt.title('Layer: '+str(l)+', Node: '+str(node))
        plt.plot(xin.cpu().detach().numpy(), yf.cpu().detach().numpy())
        plt.show()
