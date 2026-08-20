# Feynman Data Experiments

Regression experiments on dimensionless Feynman equations using PhyKAN models and learned filter representations.

## Main Files

- `Dimensionless_Feynmann.py`: dataset and dimensionless-equation utilities.
- `Dimensionless_Feynmann_name.py`: maps experiment indices to equation names.
- `Train_Feynmann2.py`: trains a PhyKAN model for one selected equation and network depth.
- `Cmap_plots.py` and `Cmap_plots_experiment.py`: visualize learned mappings and experimental transfers.
- `LowFilterVals.npy` and `HighFilterVals.npy`: filter lookup values used during training.

`Train_Feynmann2.py` expects an integer argument. It derives the equation from `argument % 27` and the network-depth index from `argument // 27`, then loads the corresponding `.npy` dataset from `FeynmanData/`. Run it from this directory and inspect the script's constants before launching a training run.
