# --- run_3_evaluate.py ---
import config
from pipeline import DataManager, Evaluator
from model import VesselTransformer
import torch
import os
import random

def main():
    print("="*60)
    print("STEP 3: EVALUATION & PLOTTING")
    print("="*60)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: No model found at {checkpoint_path}")
        print("Run 'run_2_training.py' first.")
        return

    # 1. Prepare data manager and recover splits + normalization stats
    print("Loading dataset and reconstructing normalization stats...")
    data_manager = DataManager(
        parquet_root=config.PARQUET_OUT,
        test_size=0.15,
        val_size=0.15,
    )
    try:
        data_manager.create_data_splits()
    except FileNotFoundError:
        print("Error: Processed data not found. Run 'run_1_preprocessing.py' first.")
        return

    # Same call as in run_2_training.py
    train_loader, _, test_files = data_manager.get_dataloaders(
        history_len=config.HISTORY_LEN,
        future_len=config.FUTURE_LEN,
        batch_size=config.BATCH_SIZE,
        num_workers=0
    )

    if not test_files:
        print("No test files found. Check your PARQUET_OUT directory.")
        return

    # Get normalization stats from the training dataset
    norm_stats = (train_loader.dataset.mean, train_loader.dataset.std)

    # 2. Re-initialize model and load weights
    print("Initializing model...")
    model = VesselTransformer(
        input_features=4,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_LAYERS,
        num_decoder_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=0.1
    )

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # Support both formats: either a raw state_dict or a dict wrapper
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Here checkpoint is the raw state_dict (how your Trainer saves it)
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)
    model.eval()

    # 3. Create evaluator with the recovered normalization stats
    evaluator = Evaluator(
        model=model,
        device=config.DEVICE,
        normalization_stats=norm_stats
    )

    # 4. Pick a random file from the test set and plot prediction
    rand_idx = random.randint(0, len(test_files) - 1)
    print(f"Evaluating File #{rand_idx}: {test_files[rand_idx]}")

    evaluator.evaluate_and_plot(
        test_files=test_files,
        history_len=config.HISTORY_LEN,
        file_idx=rand_idx
    )

    print("\nDone! Check 'prediction_static.png' to see the result.")

if __name__ == "__main__":
    main()
