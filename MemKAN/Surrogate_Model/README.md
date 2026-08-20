# Memristor Surrogate Models

Models that approximate memristor device behaviour for use by MemKAN experiments.

- `ModelDataPreprocess.py`: prepare device voltage, output, and parameter arrays.
- `TrainMemdiodeMLP.py`: train an MLP surrogate for the memdiode model.
- `TrainMemristor.py`: train and evaluate MLP models across hidden-layer sizes (hyperparameter validation).

These scripts expect local NumPy datasets and use CUDA by default. Check the input filenames and split sizes before running a long training job.
