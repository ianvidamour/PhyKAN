# PhyKAN

Research code for physics-aware Kolmogorov-Arnold networks (PhyKAN), including filter-based models, reinforcement learning, robotic kinematics, symbolic-regression-style Feynman experiments, memristor implementations, and photovoltaic control.

## Repository Areas

- [`Six_axis_Robot_Noised/`](6axis_Robot_Noised/README.md): noisy six-axis robot forward and inverse kinematics.
- [`Continuous_CartPole/`](Continuous_CartPole/README.md): continuous-action CartPole reinforcement learning.
- [`Feynman_Data/`](Feynman_Data/README.md): Feynman equation regression experiments.
- [`MemKAN/`](MemKAN/README.md): memristor-based KAN experiments.
- [`PV_Control/`](PV_Control/README.md): photovoltaic power-control reinforcement learning.
- [`Legacy/`](Legacy/README.md): older experiments and notebooks retained for reference.

## Running Experiments

Most scripts are research scripts with configuration values defined near the top of the file. Run them from the directory containing the script because many load `.npy` files using relative paths. Several training scripts accept an integer experiment index as their first command-line argument; inspect the script before launching a sweep.

The code uses Python, NumPy, PyTorch, and Matplotlib. Some experiments additionally use CUDA, Polars, or tqdm. There is no single repository-wide environment file, so install dependencies according to the experiment you intend to run.

## Outputs

Training and plotting scripts commonly read and write NumPy arrays, PyTorch `.pt` model files, and Matplotlib figures. Generated outputs are generally experiment-specific and are not interchangeable between folders.
