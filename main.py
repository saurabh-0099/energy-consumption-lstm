"""
Runs the full pipeline end-to-end:
  1. Preprocess raw data
  2. Train the LSTM model
  3. Evaluate on the test set
  4. Predict the next hour's consumption

Usage:
    python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import build_processed_dataset
import train
import evaluate
import predict


def run_pipeline():
    print("=" * 60)
    print("STEP 1/4: Preprocessing raw data")
    print("=" * 60)
    build_processed_dataset()

    print("\n" + "=" * 60)
    print("STEP 2/4: Training LSTM model")
    print("=" * 60)
    train.main()

    print("\n" + "=" * 60)
    print("STEP 3/4: Evaluating on test set")
    print("=" * 60)
    evaluate.main()

    print("\n" + "=" * 60)
    print("STEP 4/4: Predicting next hour's consumption")
    print("=" * 60)
    predict.predict_next_hour()

    print("\nPipeline complete! Check the 'outputs/' folder for plots and metrics.")


if __name__ == "__main__":
    run_pipeline()
