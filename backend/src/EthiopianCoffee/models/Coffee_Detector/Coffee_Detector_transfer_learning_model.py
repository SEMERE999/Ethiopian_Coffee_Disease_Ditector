import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class TransferLearningModel:
    """
    EfficientNetB0 transfer-learning model for binary coffee-leaf detection.

    Architecture:
        Augmentation -> EfficientNetB0 (frozen initially) -> GAP -> Dropout(0.40)
        -> Dense(1, sigmoid)

    EfficientNetB0 from tf.keras.applications performs its own internal preprocessing
    (pixel values expected in [0, 255] range), so no extra rescaling layer is added.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE,
        image_channels: int = training_pipeline.IMAGE_CHANNELS,
    ) -> None:
        self.image_size = image_size
        self.image_channels = image_channels

    def build_model(self) -> tf.keras.Model:
        """Build and return the EfficientNetB0 binary model."""
        try:
            logging.info("Building Detector EfficientNetB0 transfer-learning model.")

            input_shape = (
                self.image_size[0],
                self.image_size[1],
                self.image_channels,
            )

            # Pretrained base model (frozen initially - Phase 1)
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
            )
            base_model.trainable = False

            inputs = tf.keras.Input(shape=input_shape, name="image_input")

            # Data augmentation
            x = tf.keras.layers.RandomFlip("horizontal", name="random_flip")(inputs)
            x = tf.keras.layers.RandomRotation(factor=0.08, name="random_rotation")(x)
            x = tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10, name="random_zoom")(x)
            x = tf.keras.layers.RandomContrast(factor=0.10, name="random_contrast")(x)

            # EfficientNetB0 feature extraction (training=False keeps BN layers in inference mode)
            x = base_model(x, training=False)

            # Feature aggregation
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

            # Classification head
            x = tf.keras.layers.Dropout(0.40, name="dropout")(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

            model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=training_pipeline.DETECTOR_TRANSFER_LEARNING_MODEL_NAME,
            )

            logging.info("Detector EfficientNetB0 model built successfully.")
            return model

        except Exception as e:
            logging.error("Error building Detector EfficientNetB0 model.")
            raise EthiopianCoffeeException(e, sys) from e
