# MemKAN Experiments

MemKAN experiments replace the filter response used by PhyKAN-style models with memristor devices based upon a dynamic memdiode model. The folder contains related kinematics, control, and surrogate-model studies to reproduce panels similar to figures 2, 3, and 4 in the paper.

## Subprojects

- [`6axis_Robot_Kinematics/`](6axis_Robot_Kinematics/README.md): MemKAN six-axis forward kinematics.
- [`Continuous_CartPole/`](Continuous_CartPole/README.md): MemKAN CartPole results and comparisons.
- [`PV_Control/`](PV_Control/README.md): MemKAN photovoltaic control experiments.
- [`Surrogate_Model/`](Surrogate_Model/README.md): MLP and preprocessing models for emulating device behaviour.

Most scripts use CUDA, load local NumPy datasets or saved PyTorch models, and write outputs beside the script. Run each experiment from its own subdirectory and verify the expected data and model files before training.

The 'MemdiodeMLP...pt' file is the specific trained surrogate model used to generate results for all child folders.
