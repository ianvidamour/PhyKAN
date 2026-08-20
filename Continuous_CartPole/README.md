# Continuous CartPole

Continuous-action CartPole reinforcement-learning experiments using PhyKAN actor and critic networks, with MLP comparisons.

## Main Files

- `CartPole_env.py`: CartPole environment implementation.
- `PhyKAN_Util_actorcritic.py`: actor and critic model utilities.
- `KAN_CartPole_continuous.py`: PhyKAN actor-critic training loop.
- `NN_CartPole_continuous.py`: neural-network baseline.
- `PlotResults.py` and `PlotResults_2L.py`: compare trial lengths and model variants.
- `Interpolation_model_cartpole.py` and `ScaleFilters.py`: inspect learned filter behaviour.

The training scripts use `LowFilterVals.npy` and `HighFilterVals.npy` as reference hardware look-up data, and save experiment results in the working directory. Run them from this folder because the scripts use relative paths. Training is configured directly in the scripts and may require substantial CPU or GPU time.
