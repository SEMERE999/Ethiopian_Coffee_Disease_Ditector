import os
import shutil
import sys

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DiseaseDataIngestionConfig
from EthiopianCoffee.entity.artifact_entity import DiseaseDataIngestionArtifact
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class DataIngestion:
    """
    Data Ingestion component for the Coffee Leaf Disease Classification pipeline.

    Responsibilities:
    1. Locate the source train/test image directories.
    2. Verify that required source directories exist.
    3. Copy the dataset into the pipeline's artifact directory.
    4. Count the available images.
    5. Return a DiseaseDataIngestionArtifact.
    """

    def __init__(self, data_ingestion_config: DiseaseDataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config

    def _count_images(self, directory: str) -> int:
        count = 0
        for _, _, files in os.walk(directory):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in training_pipeline.SUPPORTED_IMAGE_EXTENSIONS:
                    count += 1
        return count

    def _copy_dataset(self, source_dir: str, destination_dir: str) -> None:
        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir)
        shutil.copytree(source_dir, destination_dir)

    def initiate_data_ingestion(self) -> DiseaseDataIngestionArtifact:
        logging.info("Starting Disease Data Ingestion.")
        try:
            train_source_dir = self.data_ingestion_config.train_source_dir
            test_source_dir = self.data_ingestion_config.test_source_dir
            ingested_data_dir = self.data_ingestion_config.ingested_data_dir

            if not os.path.isdir(train_source_dir):
                raise FileNotFoundError(
                    f"Training dataset directory does not exist: {train_source_dir}"
                )
            if not os.path.isdir(test_source_dir):
                raise FileNotFoundError(
                    f"Testing dataset directory does not exist: {test_source_dir}"
                )

            logging.info(f"Disease training source dir: {train_source_dir}")
            logging.info(f"Disease testing source dir: {test_source_dir}")

            os.makedirs(ingested_data_dir, exist_ok=True)

            ingested_train_dir = os.path.join(
                ingested_data_dir, training_pipeline.TRAIN_DIR_NAME
            )
            ingested_test_dir = os.path.join(
                ingested_data_dir, training_pipeline.TEST_DIR_NAME
            )

            logging.info("Copying training dataset into artifact directory.")
            self._copy_dataset(train_source_dir, ingested_train_dir)

            logging.info("Copying testing dataset into artifact directory.")
            self._copy_dataset(test_source_dir, ingested_test_dir)

            train_image_count = self._count_images(ingested_train_dir)
            test_image_count = self._count_images(ingested_test_dir)

            if train_image_count == 0:
                raise ValueError("No supported training images were found in the disease dataset.")
            if test_image_count == 0:
                raise ValueError("No supported testing images were found in the disease dataset.")

            logging.info(f"Disease training images found: {train_image_count}")
            logging.info(f"Disease testing images found: {test_image_count}")

            artifact = DiseaseDataIngestionArtifact(
                train_data_dir=ingested_train_dir,
                test_data_dir=ingested_test_dir,
                train_image_count=train_image_count,
                test_image_count=test_image_count,
                class_names=training_pipeline.DISEASE_CLASS_NAMES,
            )

            logging.info(f"Disease Data Ingestion artifact: {artifact}")
            logging.info("Disease Data Ingestion completed successfully.")
            return artifact

        except Exception as e:
            logging.error("Error occurred during Disease Data Ingestion.")
            raise EthiopianCoffeeException(e, sys) from e
