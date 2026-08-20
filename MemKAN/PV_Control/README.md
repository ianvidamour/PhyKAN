# MemKAN Photovoltaic Control

MemKAN photovoltaic-control experiments and score processing.

- `KAN_PV_Control_continuous_HPC.py`: train the MemKAN controller.
- `Process_PV_scores_KAN.py`: process saved controller scores.

The scripts load photovoltaic state, power, and current arrays, along with filter or device-model data. They are configured for CUDA and use relative paths, so run them from this directory after checking the expected input files and saved model dependencies.
