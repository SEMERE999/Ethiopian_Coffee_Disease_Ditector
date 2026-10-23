import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class TransferLearningModel:
    """
    Builds a transfer-learning model using EfficientNetB0
    pretrained on ImageNet for coffee leaf disease classification.
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
        """Build and return the EfficientNetB0 multiclass model."""
        try:
            logging.info("Building Disease EfficientNetB0 transfer-learning model.")

            input_shape = (
                self.image_size[0],
                self.image_size[1],
                self.image_channels,
            )

            # Pretrained base model (frozen initially)
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
            )
            base_model.trainable = False

            inputs = tf.keras.Input(shape=input_shape, name="image_input")

            # Data augmentation
            x = tf.keras.layers.RandomFlip("horizontal", name="random_flip")(inputs)
            x = tf.keras.layers.RandomRotation(factor=0.05, name="random_rotation")(x)
            x = tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, name="random_zoom")(x)
            x = tf.keras.layers.RandomContrast(factor=0.10, name="random_contrast")(x)

            # Feature extraction (training=False keeps BN layers in inference mode)
            x = base_model(x, training=False)

            # Feature aggregation
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

            # Classification head
            x = tf.keras.layers.Dropout(rate=0.40, name="dropout")(x)
            outputs = tf.keras.layers.Dense(units=self.num_classes, activation="softmax", name="classifier")(x)

            model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=training_pipeline.DISEASE_TRANSFER_LEARNING_MODEL_NAME,
            )

            logging.info("Disease EfficientNetB0 model built successfully.")
            return model

        except Exception as e:
            logging.error("Error occurred while building Disease EfficientNetB0 model.")
            raise EthiopianCoffeeException(e, sys) from e
