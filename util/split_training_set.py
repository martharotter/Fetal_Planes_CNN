import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def split_training_data(metadata_df: pd.core.frame.DataFrame, create_new:bool=False):
    """Split the dataset into training, testing, and validation sets"""
    # Check for existing splits
    current_dir = os.getcwd()
    data_dir = "FETAL_PLANES_ZENODO/"
    train_file = os.path.join(current_dir, data_dir, "train_split.csv")
    val_file = os.path.join(current_dir, data_dir, "val_split.csv")
    test_file = os.path.join(current_dir, data_dir, "test_split.csv")

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file) and create_new==False:
        log.info("Loading existing data splits from CSV files")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        print("Loaded existing splits")
    else:
        log.info("Creating new data splits")
        print(metadata_df.head())
        print(f"size of entire set is {len(metadata_df)}")

        train_df = metadata_df[metadata_df['Train'] == 1]
        test_df = metadata_df[metadata_df['Train'] == 0]
        print(f"size of training set is {len(train_df)}")
        print(f"size of test set is {len(test_df)}")

        # Split train into train and validation (15% for validation)
        train_df, val_df = train_test_split(
            train_df, test_size=0.15, random_state=42, stratify=train_df['Plane']
        )

        # Save new splits
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        log.info("New data splits saved to CSV files")

    print("Dataset split completed")
    print(f"Training size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    log.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df
