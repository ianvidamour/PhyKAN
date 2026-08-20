# Photovoltaic Control

Continuous-action reinforcement-learning experiments for controlling photovoltaic operating conditions with PhyKAN actor-critic models and MLP baselines.

## Main Files

- `KAN_PV_Control_continuous_HPC.py`: train PhyKAN actor and critic networks.
- `MLP_PV_Control_continuous_HPC.py`: train the MLP comparison model.
- `Process_PV_scores_KAN.py` and `Process_PV_scores_MLP.py`: process saved training scores.
- `Plot_PV.py`: plot processed KAN, MLP, and experimental results.
- `Interpolation_model_PV.py`: inspect learned filter transfer behaviour.
- `PhyKAN_Util_actorcritic.py`: actor-critic model utilities.

The training scripts load `state_data.npy`, `power_data.npy`, `current_data.npy`, `LowFilterVals.npy`, and `HighFilterVals.npy` as reference state data for the reinforcement learning environment, and the look-up tables for experimental hardware respectively. The training scripts accept an integer experiment index as an input argument that selects a network size and run number. Run from this directory, and make sure the required data files and a compatible PyTorch/CUDA environment are available.
