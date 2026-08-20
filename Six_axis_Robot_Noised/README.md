# Noisy Six-Axis Robot Kinematics

This folder evaluates PhyKAN and MLP models for six-axis robot forward and inverse kinematics under measurement noise.

## Data

`Input_Data/` contains NumPy datasets for pre-noise joint angles, noisy joint angles, true end-effector locations, and noisy end-effector locations. The files are grouped by noise level (`k=0.01`, `0.05`, and `0.1`). `LowFilterVals.npy` and `HighFilterVals.npy` provide the filter lookup values used by the PhyKAN models.

## Main Scripts

- `ForwardKinematics_HPC.py` and `ForwardKinematics_HPC_2L.py`: train one- and two-layer PhyKAN forward models.
- `InverseKinematics_HPC.py` and `InverseKinematics_HPC_2L.py`: train one- and two-layer PhyKAN inverse models.
- `ForwardKinematics_MLP*.py` and `InverseKinematics_HPC_MLP*.py`: MLP baselines.
- `Dimensionality Test *.py`: compare network dimensionality across tasks.
- `Plot_results*.py` and `Plot_edge_results*.py`: inspect accuracy and learned edge behavior.
- `Interpolation_model_*.py` and `ScaleFilters*.py`: scale experimentally transferred filter responses and provide rollout for evaluation based on experimental measurements.

Run scripts from this directory so their relative data paths resolve correctly. Check the `noise_k`, network-size, and run settings in each script before starting a long training job.
