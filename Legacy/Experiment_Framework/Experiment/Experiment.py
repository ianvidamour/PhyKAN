import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Generator
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import random
from itertools import count
from statistics import mean

import yaml    
import tqdm as tqdm
import polars as pl
import numpy as np
import pickle 
from sklearn.decomposition import PCA


from Experiment.Classes.KanWrapper import KanWrapper
from Experiment.Classes.PhyKanWrapper import PhyKAN
from Experiment.Classes.IDClass import whiten_pca_np, intrinsic_dimension, whiten_torch, safe_svd_whiten, safe_whiten_pca_np, svd_whiten
from Experiment.Classes.MLP import CustomMLP

class Experiment:
    def __init__(self, configPath: str, configFile: str, output: str, seed=0):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = 'cpu'
        torch.set_default_device(self.device)

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
        self.learning_rate = self.config["learning_rate"]

        self.l2_loss = self.config.get("l2_loss", False)

        self.refine = self.config.get('refine', False)
        self.batchnorm = self.config.get("batchnorm", False)
        
        if self.network == "phykan":
            self.filters_per_unit = self.config["filters_per_unit"]
            self.thresholding = self.config["thresholding"]
            self.lr_decay = self.config["lr_decay"]
            self.sigmoid_s = self.config["sigmoid_s"]
            self.lanbda = self.config.get("lambda", 1e-4)
            self.penalty = False if self.lanbda == 0.0 else True

            self.net = PhyKAN(self.layers, self.filters_per_unit,0,0, lr=self.learning_rate, batch_norm=self.batchnorm)
            self.schema = {
                "epoch": pl.Int8,
                "steps": pl.Int32,
                "accuracy": pl.Float32,
                "running_loss": pl.Float32,
                "loss": pl.Float32,
                "penalty_loss": pl.Float32,
                }
        else:
            self.schema = {
                "epoch": pl.Int8,
                "steps": pl.Int32,
                "accuracy": pl.Float32,
                "running_loss": pl.Float32,
                "loss": pl.Float32,
                }
            # Create Kan, MLP or PhyKan
            if self.network == "kan":
                self.grid = self.config["grid"]
                self.polynomial = self.config['polynomial']
                self.refine = self.config["refine"] 
                self.net = KanWrapper(layers=self.layers, num=self.grid, k=self.polynomial, device=self.device, batch=self.batchnorm)
                print(self.net)

            elif self.network == "mlp":
                self.net = CustomMLP(self.layers, device=self.device, batch=self.batchnorm)
            self.optimizer = torch.optim.Adam( self.net.parameters(), lr=self.learning_rate)
        
        self.criterion = torch.nn.CrossEntropyLoss()

        # Data load & transform
        #transforms.Normalize((0.5,), (0.5,)),
        train_transform = transforms.Compose([
            transforms.ToTensor(),
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
            self.mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform, target_transform=target_transform) #, target_transform=target_transform)

        generator = Generator(device=self.device)

        # Create validation set& training
        self.mnist_trainset, self.mnist_valset = torch.utils.data.random_split(self.mnist_trainset, [5000, 55000], generator=generator)
        
        #self.mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=train_transform, target_transform=target_transform)
        
        # Hyperparameters
        self.epochs = self.config["epochs"]
        self.val_epochs = 1
        
        self.batch_size = self.config['batch_size']
        self.val_batch_size =  self.config['val_batch_size']
        
        self.train_loader = DataLoader(self.mnist_trainset, shuffle=True, batch_size=self.batch_size, generator=generator)
        #self.test_loader = DataLoader(self.mnist_testset, shuffle=False, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.mnist_valset, shuffle=True, batch_size=self.val_batch_size, generator=generator)

        

        # Add a metric and output shape column for each layer
        for layer_index in range(len(self.layers) - 1):
            self.schema[f"layer_{layer_index}_dimensionality"] = pl.Float64

        # Initialize the DataFrame with the specified columns
        self.train_metrics_df = []
        self.val_metrics_df = []

        self.val_step = 0
        self.running_loss = 0
        self.val_epoch = 0

    def run(self, repeat):

        if self.network == "phykan":
            self.pk_trainingLoop(repeat)
        else:
            self.trainingLoop(repeat)

    def pk_trainingLoop(self, repeat):
        step = 0
        running_loss = 0.0
        for epoch in range(self.epochs):
            
            torch.cuda.empty_cache()

            for batch_idx, (batch_images, batch_labels) in enumerate(self.train_loader):
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
                
                #if step % 4 == 0:
                #    self.net.eval()
                #    self.pk_validationLoop(epoch)
                #    self.net.train()
                    

                #self.net.eval()
                #self.validationLoop(epoch)
                #self.net.train()
                #if step == 2000 and self.refine == True:
                #    self.net.refine(10)
                
                #import pdb; pdb.set_trace()
                outputs, loss_np, loss_p, stability_loss = self.net.train(batch_images, batch_labels, penalty=self.penalty, lamda=self.lanbda)
                #breakpoint()
                predictions = outputs[-1]
                
                running_loss += loss_np
                
                if batch_idx % 10 == 0: #in np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(batch_idx),114)))): #% 100 == 0: self.record_at=np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(N_batch),114)))) 
                    
                    predicted_indices = torch.argmax(predictions, dim=1)
                    batch_indices = torch.argmax(batch_labels, dim=1)

                    correct = (predicted_indices == batch_indices).sum().item()
                    total = batch_labels.size(0)
                    accuracy = correct / total
                    
                    output_arr = [out.cpu().detach().numpy() for out in outputs]

                    for arr in output_arr:
                        if np.isnan(arr).any():
                            print(f"NaN found in: {arr}")

                    whitened = [whiten_pca_np(out, variance_explained=0.95) for out in output_arr]
                    intrin_dims = [intrinsic_dimension(out) for out in whitened]
                    
                    row = [epoch, step, accuracy, running_loss, loss_np, loss_p] + intrin_dims
                    
                    self.train_metrics_df.append(row)

                step +=1

        exportTraining = pl.DataFrame(self.train_metrics_df, schema=self.schema, orient="row")
        exportTraining.write_csv(self.training_output + str(repeat))
        
        #exportValidation = pl.DataFrame(self.val_metrics_df, schema=self.schema, orient="row")
        #exportValidation.write_csv(self.validation_output + str(repeat))

    def pk_validationLoop(self, trainingEpoch):

        batch_metrics = {
            'accuracy': [],
            'loss_np': [],
            'loss_total': []
        }
        # Initialize list to hold intrinsic dims for each layer output
        num_expected_ids = len(self.layers) - 1
        intrin_dims_per_batch = []

        with torch.no_grad():
            for batch_idx, (batch_images, batch_labels) in enumerate(self.val_loader):
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)

                outputs = self.net.forward(batch_images, discrete=False)
                predictions = outputs[-1]

                loss_np = self.net.lossfn(predictions, batch_labels).item()
                penalty = self.net.penalty(predictions).item()

                if self.penalty == True:
                    self.running_loss += loss_np + penalty
                else:
                    self.running_loss += loss_np

                predicted_indices = torch.argmax(predictions, dim=1)
                batch_indices = torch.argmax(batch_labels, dim=1)

                correct = (predicted_indices == batch_indices).sum().item()
                total = batch_labels.size(0)
                accuracy = correct / total

                output_arr = [out.cpu().detach().numpy() for out in outputs]
                whitened = [whiten_pca_np(out, variance_explained=0.95) for out in output_arr]
                intrin_dims = [intrinsic_dimension(out) for out in whitened]

                # Verify we got the expected number of IDs
                assert len(intrin_dims) == num_expected_ids, \
                    f"Expected {num_expected_ids} intrinsic dimensions but got {len(intrin_dims)}"

                # Collect metrics for this batch
                batch_metrics['accuracy'].append(accuracy)
                batch_metrics['loss_np'].append(loss_np)
                batch_metrics['loss_total'].append(loss_np + penalty)
                intrin_dims_per_batch.append(intrin_dims)

            self.val_step += 1

            # Aggregate metrics across all batches
            avg_accuracy = np.mean(batch_metrics['accuracy'])
            avg_loss_np = np.mean(batch_metrics['loss_np'])
            avg_loss_total = np.mean(batch_metrics['loss_total'])

            # Average intrinsic dimensions across batches for each layer
            intrin_dims_array = np.array(intrin_dims_per_batch)  # Shape: (num_batches, num_layers-1)
            avg_intrin_dims = np.mean(intrin_dims_array, axis=0).tolist()

            # Create epoch summary row
            epoch_row = [self.val_epoch, self.val_step, avg_accuracy, self.running_loss, avg_loss_np, avg_loss_total] + avg_intrin_dims

            self.val_metrics_df.append(epoch_row)

    def trainingLoop(self, repeat):
        step = 0
        running_loss = 0.0

        for epoch in range(self.epochs):
            self.net.train()
        
            for batch_idx, (batch_images, batch_labels) in enumerate(self.train_loader):
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
                #breakpoint()
                
                if step % 10 == 0:
                    self.net.eval()
                    self.validationLoop(epoch)
                    self.net.train()

                if step == 2000 and self.refine == True:
                    self.net.refine(10)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(batch_images)
                
                predictions = outputs[-1]
                #breakpoint()
                #Calculate Loss
                loss = self.criterion(predictions, batch_labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if batch_idx % 10 == 0: #in np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(batch_idx),114)))): #% 100 == 0: self.record_at=np.unique(np.int32(np.exp(np.linspace(np.log(1),np.log(N_batch),114)))) 
                    
                    # Calculate accuracy
                    predicted_indices = torch.argmax(predictions.data, 1)
                    batch_indices = torch.argmax(batch_labels.data, 1) 

                    correct = (predicted_indices == batch_indices).sum().item()
                    total = batch_labels.size(0)
                    accuracy = correct / total

                    output_arr = [out.cpu().detach().numpy() for out in outputs]
                 
                    whitened = [whiten_pca_np(out) for out in output_arr]
                    #whitened = [PCA(whiten=True).fit_transform(out) for out in output_arr]

                    intrin_dims = [ 0.0 if out.shape[1]==1 else intrinsic_dimension(out) for out in whitened]
                    row = [epoch, step, accuracy, running_loss, loss] + intrin_dims
                    
                    self.train_metrics_df.append(row)

                step +=1

        exportTraining = pl.DataFrame(self.train_metrics_df, schema=self.schema, orient="row")
        exportTraining.write_csv(self.training_output + str(repeat))
        
        exportValidation = pl.DataFrame(self.val_metrics_df, schema=self.schema, orient="row")
        exportValidation.write_csv(self.validation_output + str(repeat))

    def validationLoop(self, trainingEpoch):
        with torch.no_grad():
            for epoch in range(self.val_epochs):
            
                running_loss = 0.0
                for batch_idx, (batch_images, batch_labels) in enumerate(self.val_loader):
                    batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
    
                    # Forward pass
                    outputs = self.net(batch_images)
                    predictions = outputs[-1]
    
                    #Calculate Loss
                    loss = self.criterion(predictions, batch_labels)
    
                    running_loss += loss.item()
    
                    # Calculate accuracy
                    predicted_indices = torch.argmax(predictions.data, 1)
                    batch_indices = torch.argmax(batch_labels.data, 1) 
    
                    correct = (predicted_indices == batch_indices).sum().item()
                    total = batch_labels.size(0)
                    accuracy = correct / total
                    
                    # (Rows, Features) -> (Batch size, layer size)
                    #.reshape(self.batch_size, self.layers[i + 1])
                    #print([ a.shape for a in output_arr ])
                    output_arr = [out.cpu().detach().numpy() for out in outputs]
                    whitened = [whiten_pca_np(out) for out in output_arr]
                    
                    #whitened = [PCA(whiten=True).fit_transform(out) for out in output_arr]
                    intrin_dims = [ 0.0 if out.shape[1]==1 else intrinsic_dimension(out) for out in whitened]
                    
                    row = [trainingEpoch, self.val_step, accuracy, running_loss, loss] + intrin_dims
                    
                    self.val_step +=1
    
                    self.val_metrics_df.append(row)