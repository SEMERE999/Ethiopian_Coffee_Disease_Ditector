import os
import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DetectorDataTransformationConfig
from EthiopianCoffee.entity.artifact_entity import (
    DetectorDataTransformationArtifact,
    DetectorDataValidationArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class DataTransformation:
    """
    Data Transformation component for the Coffee Leaf Detector.

    Responsibilities:
    1. Consume the DetectorDataValidationArtifact (must have passed).
    2. Verify all three split directories exist.
    3. Construct TensorFlow datasets for train, validation, and test
       to verify they can be consumed correctly.
    4. Return a DetectorDataTransformationArtifact with paths and config.
    """

    def __init__(self, data_transformation_config: DetectorDataTransformationConfig) -> None:
        self.config = data_transformation_config

    def _validate_input_directories(self, train_dir: str, val_dir: str, test_dir: str) -> None:
        for name, path in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{name} directory does not exist: {path}")

    def _build_and_verify_datasets(self, train_dir: str, val_dir: str, test_dir: str) -> None:
        image_size = self.config.image_size
        batch_size = self.config.batch_size
        class_names = training_pipeline.DETECTOR_CLASS_NAMES

        expected_image_shape = (
            image_size[0],
            image_size[1],
            training_pipeline.IMAGE_CHANNELS,
        )

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=image_size,
            color_mode="rgb",
            batch_size=batch_size,
            shuffle=True,
            seed=self.config.seed,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=image_size,
            color_mode="rgb",
            batch_size=batch_size,
            shuffle=False,
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=image_size,
            color_mode="rgb",
            batch_size=batch_size,
            shuffle=False,
        )

        for ds_name, dataset in [("training", train_ds), ("validation", val_ds), ("test", test_ds)]:
            images, labels = next(iter(dataset))
            actual_shape = tuple(images.shape[1:])
            if actual_shape != expected_image_shape:
                raise ValueError(
                    f"{ds_name} dataset has unexpected image shape. "
                    f"Expected {expected_image_shape}, got {actual_shape}."
                )

    def initiate_data_transformation(
        self, data_validation_artifact: DetectorDataValidationArtifact
    ) -> DetectorDataTransformationArtifact:
        logging.info("Starting Detector Data Transformation.")
        try:
            if not data_validation_artifact.validation_status:
                raise ValueError("Cannot proceed with data transformation: data validation failed.")

            train_dir = data_validation_artifact.train_data_dir
            val_dir = data_validation_artifact.validation_data_dir
            test_dir = data_validation_artifact.test_data_dir

            self._validate_input_directories(train_dir, val_dir, test_dir)
            self._build_and_verify_datasets(train_dir, val_dir, test_dir)

            artifact = DetectorDataTransformationArtifact(
                train_data_dir=train_dir,
                validation_data_dir=val_dir,
                test_data_dir=test_dir,
                image_size=self.config.image_size,
                image_channels=self.config.image_channels,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
            )

            logging.info(f"Detector Data Transformation completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error("Error in Detector Data Transformation stage.")
            raise EthiopianCoffeeException(e, sys) from e
