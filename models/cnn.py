from datetime import datetime
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util.split_training_set import split_training_data
from util.plotting import plot_values_from_history
from util.prepare_images import preprocess_image
from util.logging_setup import LoggingCallback
from util.logging_setup import LOGS_FOLDER
from util.logging_setup import write_metrics

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_NAME = "cnn"

log = logging.getLogger(__name__)


class BaseCNN:
    def __init__(self, metadata_df: pd.core.frame.DataFrame):
        self.metadata_df = metadata_df
        self.num_classes = self.metadata_df['Plane'].nunique()

    def apply_hparams(self, params: dict):
        for param in params:
            setattr(self, param, params[param])
        
    def run_preprocessing(self):
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_image,
            validation_split=0.15,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        train_df, val_df, test_df = split_training_data(self.metadata_df, True)

        self.train_generator = datagen.flow_from_dataframe(
            train_df, 
            x_col="Image_path", 
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=True
        )

        self.val_generator = datagen.flow_from_dataframe(
            val_df, 
            x_col="Image_path", 
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            subset="validation",
            shuffle=False
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
        self.model = Sequential([
            Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
        
            Conv2D(128, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
        
            Conv2D(256, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
        
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

    def compile_model(self, show_summary:bool=True):
        initial_learning_rate = self.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        if show_summary:
            self.model.summary()

        return

    def train_model(self):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique ID for each run
        callbacks = [
            ModelCheckpoint(
                os.path.join(LOGS_FOLDER, self.model_name, f"best_model_{run_id}.keras"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                min_delta=0.001
            ),
            # if using this callback, must manually set learning rate to a float; 
            # cannot use this callback with a learning_rate schedule object
            # tf.keras.callbacks.ReduceLROnPlateau(
            #     monitor='val_loss',
            #     factor=0.2,
            #     patience=3,
            #     min_lr=1e-6
            # ),
            LoggingCallback()
        ]

        # Train model
        log.info(f"{self.model_name} Starting training run {run_id}")

        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=20,
            steps_per_epoch=len(self.train_generator),
            validation_steps=len(self.val_generator),
            callbacks=callbacks
        )

        from sklearn.metrics import classification_report
        y_true = self.test_generator.classes
        y_pred_probs = self.model.predict(self.test_generator)
        y_pred = y_pred_probs.argmax(axis=1)
        log.info(classification_report(y_true, y_pred, target_names=self.test_generator.class_indices.keys()))

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