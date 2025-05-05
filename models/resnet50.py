from datetime import datetime
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.cnn import BaseCNN
from util.plotting import plot_values_from_history
from util.split_training_set import split_training_data
from util.prepare_images import preprocess_image
from util.logging_setup import LoggingCallback
from util.logging_setup import LOGS_FOLDER
from util.logging_setup import write_metrics

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_NAME = "resnet50"

log = logging.getLogger(__name__)


class ResNet50CNN(BaseCNN):
    def run_preprocessing(self):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

        train_df, val_df, test_df = split_training_data(self.metadata_df, False)

        self.train_generator = datagen.flow_from_dataframe(
            train_df, 
            x_col="Image_path", 
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            subset="training"
        )

        self.val_generator = datagen.flow_from_dataframe(
            val_df, 
            x_col="Image_path", 
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            subset="validation"
        )

        self.test_generator = datagen.flow_from_dataframe(
            test_df,
            x_col="Image_path",
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=False
        )

        return

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

    def compile_model(self, show_summary:bool=True):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        if show_summary:
            self.model.summary()

        return

    def train_model(self):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        callbacks = [
            ModelCheckpoint(
                os.path.join(LOGS_FOLDER, f"best_model_{run_id}.keras"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            LoggingCallback()
        ]

        # Train model
        log.info(f"{self.model_name} Starting training run {run_id}")

        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=20,
            batch_size=self.batch_size,
            callbacks=callbacks
        )

        from sklearn.metrics import classification_report
        y_true = self.test_generator.classes
        y_pred_probs = self.model.predict(self.test_generator)
        y_pred = y_pred_probs.argmax(axis=1)
        log.info(classification_report(y_true, y_pred, target_names=self.test_generator.class_indices.keys()))

        # Unfreeze some layers for fine-tuning
        # for layer in self.model.layers[:-5]:
        #     layer.trainable = True
        # Recompile with a lower learning rate
        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #     loss='categorical_crossentropy',
         #    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        # )

        # Evaluate performance
        results = self.model.evaluate(self.val_generator)
        val_loss = results[0]
        val_acc = results[1]
        log.info(f"{self.model_name} Run {run_id} completed - Validation Accuracy: {val_acc:.2%}, Validation Loss: {val_loss:.4f}")

        test_results = self.model.evaluate(self.test_generator)
        test_loss = test_results[0]
        test_acc = test_results[1]
        log.info(f"{self.model_name} Run {run_id} - Test Accuracy: {test_acc:.2%}, Test Loss: {test_loss:.4f}")

        # Append to CSV
        write_metrics(run_id, self.model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                      val_acc, val_loss, len(history.history['val_accuracy']))

        plot_values_from_history(history, run_id, log)
        log.info(f"Ending training run {run_id}")
        return