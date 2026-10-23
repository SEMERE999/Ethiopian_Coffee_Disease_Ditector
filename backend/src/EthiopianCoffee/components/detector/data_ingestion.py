import os
import shutil
import sys

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DetectorDataIngestionConfig
from EthiopianCoffee.entity.artifact_entity import DetectorDataIngestionArtifact
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class DataIngestion:
    """
    Data Ingestion component for the Coffee Leaf Detector pipeline.

    Responsibilities:
    1. Verify that dataset directory exists.
    2. Verify that coffee/ and non_coffee/ class folders exist.
    3. Copy source images into the pipeline artifact directory.
    4. Preserve source class folder structure.
    5. Count images per class.
    6. Return a DetectorDataIngestionArtifact.
    """

    def __init__(self, data_ingestion_config: DetectorDataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config

    def _count_images(self, directory: str) -> int:
        """Count supported image files recursively."""
        count = 0
        for _, _, files in os.walk(directory):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in training_pipeline.SUPPORTED_IMAGE_EXTENSIONS:
                    count += 1
        return count

    def _copy_class_folder(self, source_class_dir: str, dest_class_dir: str) -> None:
        """Copy one source class folder into the artifact destination."""
        if os.path.exists(dest_class_dir):
            shutil.rmtree(dest_class_dir)
        shutil.copytree(source_class_dir, dest_class_dir)

    def initiate_data_ingestion(self) -> DetectorDataIngestionArtifact:
        """Execute the complete data ingestion process."""
        logging.info("Starting Detector Data Ingestion.")

        try:
            dataset_dir = self.data_ingestion_config.dataset_dir
            ingested_data_dir = self.data_ingestion_config.ingested_data_dir

            if not os.path.isdir(dataset_dir):
                raise FileNotFoundError(
                    f"Detector dataset directory does not exist: {dataset_dir}"
                )

            logging.info(f"Source detector dataset directory: {dataset_dir}")

            source_classes = list(
                training_pipeline.DETECTOR_SOURCE_CLASS_NAME_MAPPING.keys()
            )

            for source_class in source_classes:
                source_class_dir = os.path.join(dataset_dir, source_class)
                if not os.path.isdir(source_class_dir):
                    raise FileNotFoundError(
                        f"Source class directory missing: {source_class_dir}"
                    )

            logging.info(f"All detector source class directories found: {source_classes}")

            os.makedirs(ingested_data_dir, exist_ok=True)

            class_image_counts: dict[str, int] = {}

            for source_class in source_classes:
                source_class_dir = os.path.join(dataset_dir, source_class)
                dest_class_dir = os.path.join(ingested_data_dir, source_class)

                logging.info(f"Copying class '{source_class}' from {source_class_dir} to {dest_class_dir}")
                self._copy_class_folder(source_class_dir, dest_class_dir)

                image_count = self._count_images(dest_class_dir)
                class_image_counts[source_class] = image_count
                logging.info(f"  '{source_class}': {image_count} images copied.")

            total_image_count = sum(class_image_counts.values())

            if total_image_count == 0:
                raise ValueError("No supported images were found in the detector dataset.")

            logging.info(f"Total detector images ingested: {total_image_count}")
            logging.info(f"Detector class image counts: {class_image_counts}")

            artifact = DetectorDataIngestionArtifact(
                ingested_data_dir=ingested_data_dir,
                total_image_count=total_image_count,
                class_image_counts=class_image_counts,
            )

            logging.info(f"Detector Data Ingestion artifact: {artifact}")
            logging.info("Detector Data Ingestion completed successfully.")
            return artifact

        except Exception as e:
            logging.error("Error occurred during Detector Data Ingestion.")
            raise EthiopianCoffeeException(e, sys) from e
