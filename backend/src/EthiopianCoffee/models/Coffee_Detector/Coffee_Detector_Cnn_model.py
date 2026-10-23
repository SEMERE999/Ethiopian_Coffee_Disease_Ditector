import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class CNNModel:
    """
    CNN-from-scratch model for binary coffee-leaf detection.

    Architecture:
        Augmentation -> Rescaling -> 4 x Conv blocks -> GAP -> Dropout(0.40)
        -> Dense(1, sigmoid)

    The final layer uses sigmoid for binary classification:
        output >= threshold -> coffee_leaf (class index 1)
        output < threshold -> not_coffee_leaf (class index 0)
    """

    def __init__(
        self,
        image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE,
        image_channels: int = training_pipeline.IMAGE_CHANNELS,
    ) -> None:
        self.image_size = image_size
        self.image_channels = image_channels

    def build_model(self) -> tf.keras.Model:
        """Build and return the binary CNN model."""
        try:
            logging.info("Building Detector CNN model from scratch.")

            input_shape = (
                self.image_size[0],
                self.image_size[1],
                self.image_channels,
            )

            inputs = tf.keras.Input(shape=input_shape, name="image_input")

            # Data augmentation (active only during training)
            x = tf.keras.layers.RandomFlip("horizontal", name="random_flip")(inputs)
            x = tf.keras.layers.RandomRotation(factor=0.08, name="random_rotation")(x)
            x = tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, name="random_zoom")(x)
            x = tf.keras.layers.RandomContrast(factor=0.10, name="random_contrast")(x)

            # Pixel normalization [0, 255] -> [0, 1]
            x = tf.keras.layers.Rescaling(scale=1.0 / 255.0, name="rescaling")(x)

            # Convolutional Block 1 - 32 filters
            x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(x)
            x = tf.keras.layers.BatchNormalization(name="bn1")(x)
            x = tf.keras.layers.ReLU(name="relu1")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

            # Convolutional Block 2 - 64 filters
            x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
            x = tf.keras.layers.BatchNormalization(name="bn2")(x)
            x = tf.keras.layers.ReLU(name="relu2")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

            # Convolutional Block 3 - 128 filters
            x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
            x = tf.keras.layers.BatchNormalization(name="bn3")(x)
            x = tf.keras.layers.ReLU(name="relu3")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

            # Convolutional Block 4 - 256 filters
            x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
            x = tf.keras.layers.BatchNormalization(name="bn4")(x)
            x = tf.keras.layers.ReLU(name="relu4")(x)

            # Feature aggregation
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

            # Classification head
            x = tf.keras.layers.Dropout(0.40, name="dropout")(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

            model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=training_pipeline.DETECTOR_CNN_MODEL_NAME,
            )

            logging.info("Detector CNN model built successfully.")
            return model

        except Exception as e:
            logging.error("Error building Detector CNN model.")
            raise EthiopianCoffeeException(e, sys) from e
