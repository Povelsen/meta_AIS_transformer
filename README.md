# AIS Transformer Project

## Overview
This project trains a Transformer-based sequence-to-sequence model to forecast short-term vessel motion from AIS (Automatic Identification System) telemetry. The workflow covers preprocessing raw AIS CSV/ZIP files, splitting data into training/validation/test sets, running model training with per-epoch validation feedback, and evaluating on the held-out test set.

## Repository Structure
- `config.py` – Central configuration for paths, device selection, and hyperparameters.
- `run_1_preprocessing.py` – Cleans raw AIS dumps and writes partitioned Parquet segments.
- `run_2_training.py` – Loads the Parquet dataset, builds the model, and trains with validation each epoch.
- `run_3_evaluate.py` – Loads the best checkpoint and visualizes predictions on the test set.
- `pipeline.py` – Data loading, dataset definitions, train/validation/test splitting, training loop, and evaluation helpers.
- `model.py`, `encoder.py`, `decoder.py`, `model_utils.py` – Transformer architecture components.

## Setup
1. Create and activate a virtual environment (example):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies (PyTorch, data utilities, plotting):
   ```bash
   pip install torch pandas pyarrow scikit-learn matplotlib tqdm haversine
   ```
3. Adjust paths in `config.py` if your raw data or output directories differ from the defaults.

## Data Preparation
1. Place your raw AIS CSV or ZIP files under `raw_ais_data/` (or update `config.DATA_RAW`).
2. Run preprocessing to generate cleaned Parquet segments:
   ```bash
   python run_1_preprocessing.py
   ```
   The script writes partitioned Parquet files under `cleaned_test_data/parquet/` (configurable via `config.PARQUET_OUT`).

## Training Workflow
1. Start training after preprocessing:
   ```bash
   python run_2_training.py
   ```
2. The `DataManager` in `pipeline.py` automatically splits the Parquet segments into **train**, **validation**, and **test** sets (default 70/15/15). Each training epoch prints training loss, validation loss, and validation distance error in meters to the terminal.
3. The best-performing weights (lowest validation loss) are stored as `checkpoints/best_model.pt`. The checkpoint includes the model weights and the normalization statistics used during training so evaluation matches the training data distribution.
4. After training, a `training_metrics.png` plot of loss and validation error is saved to the project root.

## Model Architecture
- Encoder/Decoder Transformer with positional encoding (see `model.py`).
- Input features per timestep: Latitude, Longitude, Speed Over Ground (SOG), and Course Over Ground (COG).
- Uses custom encoder/decoder layers (`encoder.py`, `decoder.py`) with GELU activations, pre-norm blocks, multi-head attention, and a feedforward network (`dim_feedforward` hidden size).
- The decoder is autoregressive during training: the last observed point seeds the decoder and targets are shifted right (teacher forcing) for sequence prediction.

## Evaluation
1. Ensure `checkpoints/best_model.pt` exists from training.
2. Run the evaluation script to test on a random held-out trajectory and generate a static plot:
   ```bash
   python run_3_evaluate.py
   ```
3. The script reloads the saved normalization stats, evaluates on the **test set**, prints the final positional error in meters, and writes `prediction_static.png` illustrating the predicted versus actual future track.

## Key Behaviors
- Training, validation, and testing datasets are explicitly separated by the `DataManager` in `pipeline.py`.
- Validation runs **every epoch** during training, with metrics printed to the terminal.
- The best model checkpoint is saved in **`.pt` format** (`checkpoints/best_model.pt`), ensuring compatibility with PyTorch loading APIs and capturing both weights and normalization statistics.
