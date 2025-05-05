import csv
import logging
import os
import pickle
from tensorflow.keras.callbacks import Callback

LOGS_FOLDER = "logs"
CHECKPOINTS_FOLDER = "checkpoints"
METRICS_FILE = "training_metrics.csv"

log = logging.getLogger(__name__)


def setup_logging():
    """Set up logging for debugging"""
    current_dir = os.getcwd()
    logs_dir = os.path.join(current_dir, LOGS_FOLDER)
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    checkpoints_dir = os.path.join(current_dir, CHECKPOINTS_FOLDER)
    os.makedirs(checkpoints_dir, exist_ok=True)


def setup_metrics():
    """Set up metrics CSV for tracking training progress"""
    current_dir = os.getcwd()
    metrics_file = os.path.join(current_dir, LOGS_FOLDER, METRICS_FILE)
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Run_ID", "Model_Name", "Timestamp", "Val_Accuracy", "Val_Loss", "Epochs_Completed"])


def write_metrics(run_id, model_name, timestamp, val_accuracy, val_loss, epochs_completed):
    """Write metrics to the CSV file"""
    current_dir = os.getcwd()
    metrics_file = os.path.join(current_dir, LOGS_FOLDER, METRICS_FILE)

    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model_name, timestamp, val_accuracy, val_loss, epochs_completed])


def write_checkpoint(self, epoch:int):
    """Write keras checkpoint files to the checkpoints file"""
    # checkpoints_dir = os.path.join(os.getcwd, CHECKPOINTS_FOLDER, self.model_name)
    checkpoints_dir = os.path.join(os.getcwd(), CHECKPOINTS_FOLDER, "cnn")
    os.makedirs(checkpoints_dir, exist_ok=True)
    # self.model.save(os.path.join(checkpoints_dir, f"checkpoint_{self.model_name}_epoch_{epoch}"))
    self.model.save(os.path.join(checkpoints_dir, f"checkpoint_cnn_epoch_{epoch}.keras"))
    return


class LoggingCallback(Callback):
    """Custom callback for logging per epoch"""
    def on_epoch_end(self, epoch:int, logs:dict=None):
        logs = logs or {}
        log.info(f"Epoch {epoch + 1} - Train Loss: {logs.get('loss'):.4f}, "
                    f"Train Acc: {logs.get('accuracy'):.4f}, "
                    f"Val Loss: {logs.get('val_loss'):.4f}, "
                    f"Val Acc: {logs.get('val_accuracy'):.4f}")

        # Save every few epoch for easy reload / analysis
        if (epoch + 1) % 5 == 0:
            write_checkpoint(self, epoch)
