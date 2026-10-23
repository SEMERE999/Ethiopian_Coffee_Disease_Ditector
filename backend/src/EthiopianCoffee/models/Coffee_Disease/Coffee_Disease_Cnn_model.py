import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class CNNModel:
    """
    CNN-from-scratch model for Ethiopian coffee leaf disease classification.

    Architecture:
        Augmentation -> Rescaling -> 4 x Conv blocks -> GAP -> Dropout(0.40)
        -> Dense(num_classes, softmax)
    """

    def __init__(
        self,
        image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE,
        image_channels: int = training_pipeline.IMAGE_CHANNELS,
        num_classes: int = training_pipeline.DISEASE_NUM_CLASSES,
    ) -> None:
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_classes = num_classes

    def build_model(self) -> tf.keras.Model:
        """Build and return the disease multiclass CNN model."""
        try:
            logging.info("Building Disease CNN model from scratch.")

            input_shape = (
                self.image_size[0],
                self.image_size[1],
                self.image_channels,
            )

            inputs = tf.keras.Input(shape=input_shape, name="image_input")

            # Data augmentation (active during training)
            x = tf.keras.layers.RandomFlip("horizontal", name="random_flip")(inputs)
            x = tf.keras.layers.RandomRotation(factor=0.05, name="random_rotation")(x)
            x = tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, name="random_zoom")(x)
            x = tf.keras.layers.RandomContrast(factor=0.10, name="random_contrast")(x)

            # Normalize pixel values [0, 255] -> [0, 1]
            x = tf.keras.layers.Rescaling(scale=1.0 / 255.0, name="rescaling")(x)

            # Convolutional Block 1 - 32 filters
            x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(x)
            x = tf.keras.layers.BatchNormalization(name="batch_norm1")(x)
            x = tf.keras.layers.ReLU(name="relu1")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool1")(x)

            # Convolutional Block 2 - 64 filters
            x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
            x = tf.keras.layers.BatchNormalization(name="batch_norm2")(x)
            x = tf.keras.layers.ReLU(name="relu2")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool2")(x)

            # Convolutional Block 3 - 128 filters
            x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
            x = tf.keras.layers.BatchNormalization(name="batch_norm3")(x)
            x = tf.keras.layers.ReLU(name="relu3")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool3")(x)

            # Convolutional Block 4 - 256 filters
            x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
            x = tf.keras.layers.BatchNormalization(name="batch_norm4")(x)
            x = tf.keras.layers.ReLU(name="relu4")(x)

            # Feature aggregation
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

            # Classification head
            x = tf.keras.layers.Dropout(rate=0.40, name="dropout")(x)
            outputs = tf.keras.layers.Dense(units=self.num_classes, activation="softmax", name="classifier")(x)

            model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=training_pipeline.DISEASE_CNN_MODEL_NAME,
            )

            logging.info("Disease CNN model built successfully.")
            return model

        except Exception as e:
            logging.error("Error occurred while building Disease CNN model.")
            raise EthiopianCoffeeException(e, sys) from e
