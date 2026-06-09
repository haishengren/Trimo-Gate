# TriMo-Gate

Code for the TriMo-Gate molecular property prediction model. This repository provides the implementation used for the manuscript experiments on the QM9 dataset, combining three molecular representations:

- 1D SMILES token sequences
- 2D molecular graphs
- 3D molecular point clouds from QM9 `.xyz` structures

The training script follows the random-split protocol described in the manuscript: each experiment uses an 8:1:1 train/validation/test split, repeats the split with three random seeds, performs Bayesian hyperparameter optimization for each seed, retrains the final model with the selected configuration, and evaluates the held-out test set.

## Repository Structure

```text
.
|-- Train_and_evaluation.py        # Main training, Bayesian optimization, evaluation, and result export script
|-- Model_1.py                     # TriMo-Gate model components and multimodal fusion modules
|-- Graph_dataloader.py            # 2D molecular graph construction from SMILES with RDKit
|-- SMILES_dataloader_atom.py      # SMILES atom-wise tokenization and sequence indexing
`-- Ponit_cloud_dataloader.py      # 3D point-cloud construction from QM9 xyz files
```

## Model Overview

TriMo-Gate uses three modality-specific encoders:

1. **SMILES encoder**: atom-wise tokenization, token embedding, sinusoidal positional encoding, Transformer encoder layers, and 1D CNN pooling.
2. **Graph encoder**: directed message passing neural network (DMPNN) over RDKit molecular graphs.
3. **Point-cloud encoder**: PointNet-style encoder using centered 3D atomic coordinates, radial distance, and atom-type one-hot features.

The encoded representations are fused for molecular property prediction. The current implementation uses concatenation-based fusion in `MultimodalModel`; alternative weighted-sum and gated fusion modules are also implemented in `Model_1.py`.

## Requirements

The code was written in Python and depends on the following packages:

```text
numpy
pandas
matplotlib
scikit-learn
openpyxl
optuna>=3.0
torch
torch-geometric
torch-scatter
rdkit
SmilesPE
```

Example installation with conda:

```bash
conda create -n trimogate python=3.10
conda activate trimogate

conda install -c conda-forge rdkit numpy pandas matplotlib scikit-learn openpyxl
pip install optuna SmilesPE
```

Install PyTorch, PyTorch Geometric, and `torch-scatter` according to your CUDA/PyTorch version. Please follow the official installation instructions:

- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Data Preparation

Place the QM9 property table and QM9 xyz files in the repository root.

Expected files:

```text
.
|-- QM9_dataset.xlsx
|-- QM9_xyz/
|   |-- dsgdb9nsd_000001.xyz
|   |-- dsgdb9nsd_000002.xyz
|   `-- ...
```

The data loaders expect:

- `QM9_dataset.xlsx`: the first column contains SMILES strings.
- The target property is read from the second-to-last column of `QM9_dataset.xlsx`.
- `QM9_xyz/`: contains the original QM9 xyz files named as `dsgdb9nsd_000001.xyz`, `dsgdb9nsd_000002.xyz`, etc.

If your dataset file or target column is different, update the corresponding lines in:

- `Graph_dataloader.py`
- `SMILES_dataloader_atom.py`
- `Ponit_cloud_dataloader.py`

## Important Notes Before Running

There are two filename/import details to check before running the current code:

1. `Train_and_evaluation.py` imports `Point_cloud_dataloader`, while the repository file is currently named `Ponit_cloud_dataloader.py`. Rename the file to `Point_cloud_dataloader.py`, or update the import statement.
2. `Ponit_cloud_dataloader.py` uses `XYZ_DIR`, but the variable is currently commented out. Set it to the folder containing the QM9 xyz files, for example:

```python
XYZ_DIR = r"QM9_xyz"
```

The Bayesian optimization function also uses Optuna, so make sure `optuna` is imported in `Train_and_evaluation.py` if it is not already present:

```python
import optuna
```

## Running the Experiment

After preparing the environment and data, run:

```bash
python Train_and_evaluation.py
```

By default, the script uses:

- Random seeds: `42`, `44`, `46`
- Train/validation/test split: `0.8/0.1/0.1`
- Bayesian optimization trials per seed: `15`
- Maximum Bayesian optimization epochs: `200`
- Maximum final training epochs: `200`
- Device: CUDA if available, otherwise CPU

These settings can be changed in the basic configuration section of `Train_and_evaluation.py`.

## Output Files

Results are saved to:

```text
RandomSplit_3_Seeds_1D+2D+3D_BayesianOptimization_Results/
```

Main outputs include:

```text
random_split_seed_{seed}.xlsx                  # Train/validation/test split for each seed
bayesian_optimization_trials_seed_{seed}.xlsx  # Optuna trial records
best_hyperparameters_seed_{seed}.json          # Best hyperparameters for each seed
training_history_seed_{seed}.xlsx              # Training and validation loss history
predictions_seed_{seed}.xlsx                   # Test predictions for each seed
random_3_seeds_metrics.xlsx                    # Test MAE, RMSE, and R2 for all seeds
best_hyperparameters_all_seeds.xlsx            # Best configurations across seeds
all_random_splits.xlsx                         # Combined split records
all_predictions.xlsx                           # Combined test predictions
publication_performance_summary.png/.pdf       # Summary figure for publication
publication_parity_plot.png/.pdf               # Prediction parity plot
```

## Modality Selection

The active modalities are inferred from the result directory name in `Train_and_evaluation.py`.

For example:

```python
RESULT_DIR = "RandomSplit_3_Seeds_1D+2D+3D_BayesianOptimization_Results"
```

activates all three modalities:

- `1D`: SMILES
- `2D`: graph
- `3D`: point cloud

To run ablation experiments, change the modality tokens in `RESULT_DIR`, for example:

```text
RandomSplit_3_Seeds_1D_Results
RandomSplit_3_Seeds_2D_Results
RandomSplit_3_Seeds_3D_Results
RandomSplit_3_Seeds_1D+2D_Results
RandomSplit_3_Seeds_1D+3D_Results
RandomSplit_3_Seeds_2D+3D_Results
```

## Citation

If you use this code, please cite the associated manuscript:

```bibtex
@article{trimogate,
  title   = {TriMo-Gate: Multimodal Molecular Representation Learning with SMILES, Graph, and 3D Point-Cloud Features},
  author  = {Author names},
  journal = {Journal name},
  year    = {Year}
}
```

Please replace the placeholder citation information with the final manuscript details.

## License

Please add the appropriate license for this repository before public release.
