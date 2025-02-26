import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import importlib
from itertools import count
from statistics import mean
import random
import yaml    
import tqdm as tqdm
import polars as pl
import numpy as np

from typing import Optional

from sklearn.decomposition import PCA

from Experiment.Classes.KanWrapper import KanWrapper
from Experiment.Classes.PhyKanWrapper import PhyKAN
#from Experiment.Classes.ID import intrinsic_dimension
from Experiment.Classes.IDClass import whiten_pca_np, intrinsic_dimension
from Experiment.Classes.MLP import CustomMLP

"""
        if output:
            self.training_output = "./Data/" + output + "/train_" + str(repeatNumber)
            self.validation_output = "./Data/" + output + "/val_" + str(repeatNumber)
        else:      
"""

class Experiment:
    def __init__(self, configPath: str, configFile: str, output: str):

        self.device = 'cpu' #torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")

        self.training_output = output + "/_train_" 
        self.validation_output = output + "/_val_" 

        # Open config YAML and load to instance attributes 
        with open(configPath + configFile  + '.yaml', 'r') as file:
            try:
                content = file.read()
                self.config = {}
                self.config = yaml.safe_load(content)
            except FileNotFoundError:
                print(f"Error: The file {configPath} was not found.")
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")

        # Initalise self.network
        self.network = self.config["network"]
        self.layers = self.config["layers"]
        
        self.refine = False
        
        # Create Kan or MLP
        if self.network == "kan":
            self.grid = self.config["grid"]
            self.polynomial = self.config['polynomial']
            self.refine = self.config["refine"] 
            self.net = KanWrapper(layers=self.layers, num=self.grid, k=self.polynomial, device= self.device)
        elif self.network == "mlp":
            self.net = CustomMLP(self.layers, device=self.device)
        elif self.network == "phykan":
            self.filters_per_unit = self.config["filters_per_unit"]
            self.thresholding = self.config["thresholding"]
            self.lr_decay = self.config["lr_decay"]
            self.sigmoid_s = self.config["sigmoid_s"]
            self.net = PhyKAN(self.layers, self.filters_per_unit )

        # Data load & transform
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        
        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
            ])
        
        noisy_target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1) if torch.rand(1) > 0.1 else torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(np.random.choice([x for x in range(10) if x != y])), value=1))
            ])

        noisy_labels = transforms.Compose([
            transforms.Lambda(lambda y: torch.randint(0, 10, ()) if torch.rand(1) < 0.1 else torch.tensor(y))
            ])
        
        # Create dataset with optional noise
        if self.config["noisy"] == True:
            self.mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform, target_transform=noisy_labels) #, target_transform=target_transform)
        else:
            self.mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform) #, target_transform=target_transform)
        
        # Create validation set& training
        self.mnist_trainset, self.mnist_valset = torch.utils.data.random_split(self.mnist_trainset, [50000, 10000])
        self.mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=train_transform, target_transform=target_transform)
        
        # Hyperparameters
        self.epochs = self.config["epochs"]
        self.val_epochs = 1
        
        self.batch_size = self.config['batch_size']
        self.val_batch_size =  self.config['val_batch_size']
        
        self.train_loader = DataLoader(self.mnist_trainset, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.mnist_testset, shuffle=False, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.mnist_valset, shuffle=True, batch_size=self.val_batch_size)

        self.learning_rate = self.config["learning_rate"]
        self.optimizer = torch.optim.Adam( self.net.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.schema = {
            "epoch": pl.Int8,
            "steps": pl.Int32,
            "accuracy": pl.Float32,
            "loss": pl.Float32,
        }

        # Add a metric and output shape column for each layer
        for layer_index in range(len(self.layers) - 1):
            self.schema[f"layer_{layer_index}_dimensionality"] = pl.Float64

        # Initialize the DataFrame with the specified columns
        self.train_metrics_df = []
        self.val_metrics_df = []
        self.val_step = 0

    def run(self, repeat):
        self.trainingLoop(repeat)
        
    def trainingLoop(self, repeat):
        #pca = PCA(whiten=True)
        step = 0
        running_loss = 0.0

        #breakpoint()

        for epoch in range(self.epochs):
            self.net.train()
            #breakpoint()
            #
            for batch_idx, (batch_images, batch_labels) in enumerate(self.train_loader):
                
                if batch_idx % 10 == 0:
                    self.net.eval()
                    #self.validationLoop()
                    self.net.train()

                if step == 2000 and self.refine == True:
                    self.net.refine(10)

                self.optimizer.zero_grad()

                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)

                # Forward pass
                outputs = self.net(batch_images)
                
                predictions = outputs[-1]

                #Calculate Loss
                loss = self.criterion(predictions, batch_labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # Rewrite code only for first hidden layer to confirm steps
                
                if batch_idx % 10 == 0: #in np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(batch_idx),114)))): #% 100 == 0: self.record_at=np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(N_batch),114)))) 
                    #breakpoint()
                    #import pdb; pdb.set_trace()
                    # Calculate accuracy
                    predicted = torch.argmax(predictions.data, 1)

                    correct = (predicted == batch_labels).sum().item()
                    total = batch_labels.size(0)
                    accuracy = correct / total
                    # (Rows, Features) -> (Batch size, layer size)
                    #.reshape(self.batch_size, self.layers[i + 1])
                    #print([ a.shape for a in output_arr ])
                    #breakpoint()
                    output_arr = [out.cpu().detach().numpy() for out in outputs]
                    #whitened = [pca.fit_transform(out) for out in output_arr]
                    whitened = [whiten_pca_np(out) for out in output_arr]
                    intrin_dims = [ 0.0 if out.shape[1]==1 else intrinsic_dimension(out) for out in whitened]
                    row = [epoch, step, accuracy, running_loss] + intrin_dims
                    
                    self.train_metrics_df.append(row)
                    #breakpoint()

                step +=1

        exportTraining = pl.DataFrame(self.train_metrics_df, schema=self.schema, orient="row")
        exportTraining.write_csv(self.training_output + str(repeat))
        
        exportValidation = pl.DataFrame(self.val_metrics_df, schema=self.schema, orient="row")
        exportValidation.write_csv(self.validation_output + str(repeat))

    def validationLoop(self):
        
        for epoch in range(self.val_epochs):

            running_loss = 0.0
            for batch_idx, (batch_images, batch_labels) in enumerate(self.val_loader):
                self.optimizer.zero_grad()

                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)

                # Forward pass
                outputs = self.net(batch_images)
                predictions = outputs[-1]

                #Calculate Loss
                loss = self.criterion(predictions, batch_labels)

                running_loss += loss.item()
                # Rewrite code only for first hidden layer to confirm steps
                
                     #in np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(batch_idx),114)))): #% 100 == 0: self.record_at=np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(N_batch),114)))) 
                    #breakpoint()

                    # Calculate accuracy
                predicted = torch.argmax(predictions.data, 1)

                correct = (predicted == batch_labels).sum().item()
                total = batch_labels.size(0)
                accuracy = correct / total
                
                # (Rows, Features) -> (Batch size, layer size)
                #.reshape(self.batch_size, self.layers[i + 1])
                #print([ a.shape for a in output_arr ])
                output_arr = [out.cpu().detach().numpy() for out in outputs]
                #whitened = [pca.fit_transform(out) for out in output_arr]
                whitened = [whiten_pca_np(out) for out in output_arr]
                
                intrin_dims = [ 0.0 if out.shape[1]==1 else intrinsic_dimension(out) for out in whitened]
                
                row = [epoch, self.val_step, accuracy, running_loss] + intrin_dims
                
                self.val_step +=1

        self.val_metrics_df.append(row)

    """def validationLoop(self):
        
        for epoch in range(self.val_epochs):
            

            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            # To store intrinsic dimensions for averaging later
            epoch_intrin_dims = [0.0] * (len(self.layers) -1)
            
            self.val_step +=1
            for batch_idx, (batch_images, batch_labels) in enumerate(self.val_loader):
                
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)

                # Forward pass
                outputs = self.net(batch_images)

                predictions = outputs[-1]

                # Calculate Loss
                loss = self.criterion(predictions, batch_labels)

                # Backward pass and optimize
                #loss.backward()

                # Accumulate loss
                running_loss += loss.item()

                # Calculate accuracy
                predicted = torch.argmax(predictions.data, 1)
                correct = (predicted == batch_labels).sum().item()
                total_correct += correct
                total_samples += batch_labels.size(0)

                # Accumulate intrinsic dimensions for each layer
                for i in range(len(outputs)):
                    out = outputs[i].cpu().detach().numpy()#.reshape(batch_images.size(0), -1)
                    #print(out.shape)
                    whitened = whiten_pca_np(out) 
                    #print(whitened.shape)
                    intrinsic_dim = intrinsic_dimension(whitened)
                    #print(intrinsic_dim.shape)
                    epoch_intrin_dims[i] += intrinsic_dim

        # Calculate average metrics for the epoch
        avg_loss = running_loss / batch_idx
        avg_accuracy = total_correct / total_samples
        avg_intrin_dims = [dim / total_samples for dim in epoch_intrin_dims]

        # Prepare row to store
        row = [epoch, self.val_step, avg_accuracy, avg_loss] + avg_intrin_dims
    
        self.val_metrics_df.append(row)
"""