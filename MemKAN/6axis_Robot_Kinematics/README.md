# MemKAN Six-Axis Kinematics

Memristor-based MemKAN models for noisy six-axis robot forward kinematics.

The training scripts load pre-noise joint-angle data and noisy end-effector locations, then train models with different hidden-layer widths and run indices selected from the command-line argument. `Plot_results.py` visualizes saved predictions. The scripts expect `MemKAN_Util.py`, the relevant `.npy` datasets, and a compatible saved memdiode model in the working directory.

Run from this folder and inspect the `Netsizes`, `noise_k`, and device settings before starting a CUDA training run.
