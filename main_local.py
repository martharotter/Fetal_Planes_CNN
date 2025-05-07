import argparse
import logging
import os
import pandas as pd
from ruamel.yaml import YAML

from models.cnn import BaseCNN
from models.resnet50 import ResNet50CNN
from models.inception import InceptionCNN
from util.logging_setup import setup_logging
from util.logging_setup import setup_metrics
from build_cnn_with_efficientNet import run_efficientNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        required=False,
        help="Pass name of model, either 'cnn', 'resnet', 'efficientnet', or 'inception'"
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

    # Load hparams
    yaml_file = "hparams.yaml"
    params = {}
    with open(yaml_file) as f:
        yaml = YAML(typ='safe')
        # Load and find
        yaml_map = yaml.load(f)
        params_dict = yaml_map[args.model]

    # Load dataset
    current_dir = os.getcwd()
    data_dir = "FETAL_PLANES_ZENODO/"
    images_dir = "Images/"
    csv_name = "FETAL_PLANES_DB_data.csv"
    metadata_path = os.path.join(current_dir, data_dir, csv_name)
    image_path = os.path.join(current_dir, data_dir, images_dir)

    metadata_df = pd.read_csv(metadata_path, delimiter=";")
    metadata_df['Image_path'] = image_path + metadata_df['Image_name']  + ".png"

    if args.model == "CNN":
        model = BaseCNN(metadata_df=metadata_df)
        model.apply_hparams(params_dict)
        model.run_preprocessing()
        model.build_model()
        model.compile_model()
        model.train_model()

    elif args.model == "RESNET50":
        model = ResNet50CNN(metadata_df=metadata_df)
        model.apply_hparams(params_dict)
        model.run_preprocessing()
        model.build_model()
        model.compile_model()
        model.train_model()

    elif args.model == "efficientnet":
        run_efficientNet(metadata_df, params)

    elif args.model == "INCEPTION":
        model = InceptionCNN(metadata_df=metadata_df)
        model.apply_hparams(params_dict)
        model.run_preprocessing()
        model.build_model()
        model.compile_model()
        model.train_model()
        
    return


if __name__ == '__main__':
    main()
