import argparse
import logging
import os
import pandas as pd

from util.logging_setup import setup_logging
from util.logging_setup import setup_metrics
from build_cnn import run_cnn
from build_cnn_with_resnet50 import run_resnet50
from build_cnn_with_efficientNet import run_efficientNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        required=False,
        help="Pass name of model, either 'cnn', 'resnet', or 'efficientnet'"
        )
    args = parser.parse_args()
    setup_logging()
    setup_metrics()
    log = logging.getLogger(__name__)

    # Log GPU usage
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        log.info("Using GPU:", physical_devices)
    else:
        log.info("Using CPU")

    # Load dataset
    current_dir = os.getcwd()
    data_dir = "FETAL_PLANES_ZENODO/"
    images_dir = "Images/"
    csv_name = "FETAL_PLANES_DB_data.csv"
    metadata_path = os.path.join(current_dir, data_dir, csv_name)
    image_path = os.path.join(current_dir, data_dir, images_dir)

    metadata_df = pd.read_csv(metadata_path, delimiter=";")
    metadata_df['Image_path'] = image_path + metadata_df['Image_name']  + ".png"

    if args.model == "cnn":
        run_cnn(metadata_df)
    elif args.model == "resnet":
        run_resnet50(metadata_df)
    elif args.model == "efficientnet":
        run_efficientNet(metadata_df)
    else:  # Default run all three
        log.info("Starting run of all three models, this may take a while to run, please be patient")
        run_cnn(metadata_df)
        run_resnet50(metadata_df)
        run_efficientNet(metadata_df)
        log.info("Finished run of all three models, thanks for your patience!")
    return


if __name__ == '__main__':
    main()
