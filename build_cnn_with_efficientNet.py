from datetime import datetime
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from util.split_training_set import split_training_data
from util.prepare_images import preprocess_image
from util.logging_setup import LoggingCallback
from util.logging_setup import LOGS_FOLDER
from util.logging_setup import write_metrics

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_NAME = "efficientNet"

log = logging.getLogger(__name__)

def run_efficientNet(metadata_df):
    num_classes = metadata_df['Plane'].nunique()

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

    train_df, val_df, test_df = split_training_data(metadata_df)

    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col="Image_path",
        y_col="Plane",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col="Image_path",
        y_col="Plane",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_generator = datagen.flow_from_dataframe(
        test_df,
        x_col="Image_path",
        y_col="Plane",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    base_model = EfficientNetB0(weights='imagenet',
                                include_top=False,
                                input_shape=(224, 224, 3))

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        ModelCheckpoint(os.path.join(LOGS_FOLDER, f"best_model_{run_id}.keras"), 
                    save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        LoggingCallback()
    ]

    log.info(f"{MODEL_NAME} Starting training run {run_id}")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(val_generator)
    log.info(f"{MODEL_NAME} Run {run_id} completed - Validation Accuracy: {val_acc:.2%}, Validation Loss: {val_loss:.4f}")

    test_loss, test_acc = model.evaluate(test_generator)
    log.info(f"{MODEL_NAME} Run {run_id} - Test Accuracy: {test_acc:.2%}, Test Loss: {test_loss:.4f}")

    write_metrics(run_id, MODEL_NAME, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                  val_acc, val_loss, len(history.history['val_accuracy']))

    log.info(f"Ending training run {run_id}")
    return
