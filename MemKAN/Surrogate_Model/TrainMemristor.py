import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tqdm
import os

device = 'cuda'

X = np.load('MemdiodeModelX.npy')
Y = np.load('MemdiodeModelY.npy')

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

Nsplits = 100
Nrepeats = 2
Ntrain = 3_000_000
Nval = 500_000
Ntest = 500_000
Niter = 1_000_000
Nbatch = 10_000

# Training and validation schema
schema = {
    "N": pl.Int16,
    "repeat": pl.Int16,
    "i": pl.Int32,
    "split": pl.String,
    "loss": pl.Float64
    }

# Storing outputs for each model iteration
rows = []

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

test_file_X = "X_test.npy"
test_file_Y = "Y_test.npy"
train_val_file_X = "X_train_val.npy"
train_val_file_Y = "Y_train_val.npy"

all_files_exist = all(os.path.exists(f) for f in [
    test_file_X, test_file_Y, train_val_file_X, train_val_file_Y
])

if all_files_exist:
    # Load both splits from disk — no reliance on X/Y ordering
    X_train_val = torch.from_numpy(np.load(train_val_file_X)).float().to(device)
    Y_train_val = torch.from_numpy(np.load(train_val_file_Y)).float().to(device)
    X_test = torch.from_numpy(np.load(test_file_X)).float().to(device)
    Y_test = torch.from_numpy(np.load(test_file_Y)).float().to(device)
else:
    # First run — split X/Y and save both portions to disk
    X_train_val, X_test = torch.split(
        torch.from_numpy(X).float().to(device), [Ntrain + Nval, Ntest]
    )
    Y_train_val, Y_test = torch.split(
        torch.from_numpy(Y).float().to(device), [Ntrain + Nval, Ntest]
    )
    np.save(train_val_file_X, X_train_val.cpu().numpy())
    np.save(train_val_file_Y, Y_train_val.cpu().numpy())
    np.save(test_file_X, X_test.cpu().numpy())
    np.save(test_file_Y, Y_test.cpu().numpy())

# K-fold loop — shuffle train_val each repeat, test set is never touched
for repeat in tqdm.trange(Nrepeats):
    torch.cuda.empty_cache()

    perm = torch.randperm(Ntrain + Nval, device=device)
    X_train, X_val = torch.split(X_train_val[perm], [Ntrain, Nval])
    Y_train, Y_val = torch.split(Y_train_val[perm], [Ntrain, Nval])
    
    for N in [50, 75, 100, 125, 150, 175, 200, 250, 300, 500]:

        model = nn.Sequential(nn.Linear(6, N), nn.ReLU(), nn.Linear(N, N), nn.ReLU(), nn.Linear(N, 1)).to(device)
        model = torch.compile(model)
        optimiser = optim.Adam(params=model.parameters(), lr=3e-4)
        lossfn = nn.MSELoss()

        for i in range(Niter + 1):

            #Training steps
            Xin, Yin = gen_samples(X_train, Y_train, Nbatch)
            optimiser.zero_grad()
            pred = model.forward(Xin)
            loss = lossfn(pred, Yin)
            loss.backward()
            optimiser.step()
            
            # Validation steps
            if i % 100 == 0:
                val_running_loss = 0    
                for split in range(Nsplits):
                    pred = model.forward(X_val[split*int(Nval/Nsplits):(split+1)*int(Nval/Nsplits)])
                    loss = lossfn(pred, Y_val[split*int(Nval/Nsplits):(split+1)*int(Nval/Nsplits)])
                    val_running_loss += loss.item()/Nsplits
                
                rows.append({"N": N, "repeat": repeat, "i": i, "split": "val", "loss": val_running_loss})
                #print(N, i, val_running_loss)

        with torch.no_grad():
            test_running_loss = 0
            
            for split in range(Nsplits):
                pred = model.forward(X_test[split*int(Ntest/Nsplits):(split+1)*int(Ntest/Nsplits)])
                loss = lossfn(pred, Y_test[split*int(Ntest/Nsplits):(split+1)*int(Ntest/Nsplits)])
                test_running_loss += loss.item()/Nsplits

            rows.append({"N": N, "repeat": repeat, "i": i, "split": "test", "loss": test_running_loss})
            
            print(N, i, test_running_loss)
            print(N, 'Test Accuracy:', test_running_loss)
    
        pl.DataFrame(data=rows, schema=schema).write_parquet('./Data/3E4/_' + str(repeat) + '_' + str(N) + '.pq')
        torch.save(model,f'./Data/3E4/_model_{repeat}_{N}_.pt')
        rows = []
