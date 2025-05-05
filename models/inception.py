from datetime import datetime
import logging
import numpy as np
import pandas as pd
import os
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn import BaseCNN
from util.split_training_set import split_training_data
from util.plotting import plot_values_from_history
from util.logging_setup import LoggingCallback
from util.logging_setup import LOGS_FOLDER
from util.logging_setup import write_metrics

# Constants
IMG_SIZE = (299, 299)

log = logging.getLogger(__name__)


class InceptionCNN(BaseCNN):
    def run_preprocessing(self):
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        test_val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_df, val_df, test_df = split_training_data(self.metadata_df, True)

        # Training data generator
        self.train_generator = train_datagen.flow_from_dataframe(
            train_df, 
            x_col="Image_path", 
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=True
        )


        self.val_generator = test_val_datagen.flow_from_dataframe(
            val_df, 
            x_col="Image_path", 
            y_col="Plane",
            target_size=IMG_SIZE,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=False  # No shuffling for consistent validation
        )

        self.test_generator = test_val_datagen.flow_from_dataframe(
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
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        base_model.trainable = False

        inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)

    def get_indexed_class_weight(self):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.metadata_df['Plane']),
            y=self.metadata_df['Plane']
        )
        class_weight_dict = dict(zip(np.unique(self.metadata_df['Plane']), class_weights))

        # Map class names to indices
        class_indices = self.train_generator.class_indices
        index_to_class = {v: k for k, v in class_indices.items()}

        # Remap class_weight dict to use index keys
        indexed_class_weight = {class_indices[k]: v for k, v in class_weight_dict.items()}

        return indexed_class_weight

    def compile_model(self, show_summary:bool=True):
        initial_learning_rate = self.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

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
                os.path.join(LOGS_FOLDER, f"best_model_{run_id}.keras"),
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
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            ),
            LoggingCallback()
        ]

        indexed_class_weight = self.get_indexed_class_weight()

        # Train model
        log.info(f"{self.model_name} Starting training run {run_id}")

        history = self.model.fit(
            x=self.train_generator,
            validation_data=self.val_generator,
            epochs=20,
            steps_per_epoch=len(self.train_generator),
            validation_steps=len(self.val_generator),
            callbacks=callbacks,
            class_weight=indexed_class_weight
        )

        # answer how well each class is performing
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