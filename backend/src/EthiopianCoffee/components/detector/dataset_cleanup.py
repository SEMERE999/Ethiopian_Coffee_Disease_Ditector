import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DetectorDatasetCleanupConfig
from EthiopianCoffee.entity.artifact_entity import (
    DetectorDataIngestionArtifact,
    DetectorDatasetCleanupArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class DatasetCleanup:
    """
    Dataset Cleanup component for the Coffee Leaf Detector pipeline.

    Pipeline:
      collect images
        -> canonicalize class names
        -> SHA-256 hashing
        -> exact duplicate detection
        -> conflicting-label detection
        -> quarantine conflicts
        -> keep one representative per duplicate group
        -> stratified train / validation / test split
        -> copy clean dataset
        -> generate cleanup report
    """

    def __init__(self, dataset_cleanup_config: DetectorDatasetCleanupConfig) -> None:
        self.config = dataset_cleanup_config

    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as fh:
            while chunk := fh.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _collect_images(self, ingested_data_dir: str) -> list[dict]:
        """Collect images and map to canonical detector classes."""
        records: list[dict] = []

        if not os.path.isdir(ingested_data_dir):
            raise FileNotFoundError(
                f"Ingested data directory does not exist: {ingested_data_dir}"
            )

        for source_class in sorted(os.listdir(ingested_data_dir)):
            source_class_dir = os.path.join(ingested_data_dir, source_class)
            if not os.path.isdir(source_class_dir):
                continue

            if source_class not in training_pipeline.DETECTOR_SOURCE_CLASS_NAME_MAPPING:
                logging.warning(f"Skipping unknown source class directory: '{source_class}'")
                continue

            canonical_class = training_pipeline.DETECTOR_SOURCE_CLASS_NAME_MAPPING[source_class]

            for root, _, files in os.walk(source_class_dir):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in training_pipeline.SUPPORTED_IMAGE_EXTENSIONS:
                        continue

                    file_path = os.path.join(root, fname)
                    sha256 = self._calculate_sha256(file_path)

                    records.append({
                        "path": file_path,
                        "source_class": source_class,
                        "canonical_class": canonical_class,
                        "sha256": sha256,
                    })

        return records

    def _group_by_hash(self, records: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            groups[record["sha256"]].append(record)
        return groups

    def _resolve_duplicate_groups(
        self, hash_groups: dict[str, list[dict]]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        unique_records: list[dict] = []
        duplicate_groups: list[dict] = []
        conflicting_groups: list[dict] = []

        for sha256, records in hash_groups.items():
            classes = {r["canonical_class"] for r in records}

            if len(classes) > 1:
                conflicting_groups.append({"sha256": sha256, "records": records})
                continue

            if len(records) > 1:
                duplicate_groups.append({"sha256": sha256, "records": records})

            unique_records.append(records[0])

        return unique_records, duplicate_groups, conflicting_groups

    def _split_records(
        self, records: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        rng = random.Random(self.config.random_seed)
        records_by_class: dict[str, list[dict]] = defaultdict(list)

        for record in records:
            records_by_class[record["canonical_class"]].append(record)

        train_records: list[dict] = []
        validation_records: list[dict] = []
        test_records: list[dict] = []

        for class_name in training_pipeline.DETECTOR_CLASS_NAMES:
            class_records = records_by_class.get(class_name, [])
            if not class_records:
                raise ValueError(f"No usable images remain for class: '{class_name}'")

            rng.shuffle(class_records)
            total = len(class_records)

            train_count = int(total * self.config.train_split_ratio)
            val_count = int(total * self.config.validation_split_ratio)

            train_records.extend(class_records[:train_count])
            validation_records.extend(class_records[train_count:train_count + val_count])
            test_records.extend(class_records[train_count + val_count:])

        return train_records, validation_records, test_records

    def _copy_records(self, records: list[dict], split_name: str, destination_root: str) -> None:
        for idx, record in enumerate(records):
            class_dir = os.path.join(destination_root, split_name, record["canonical_class"])
            os.makedirs(class_dir, exist_ok=True)
            original_fname = os.path.basename(record["path"])
            dest_fname = f"{record['sha256'][:12]}_{idx:06d}_{original_fname}"
            shutil.copy2(record["path"], os.path.join(class_dir, dest_fname))

    def _quarantine_conflicting_groups(self, conflicting_groups: list[dict]) -> None:
        quarantine_dir = self.config.quarantine_dir
        for group_idx, group in enumerate(conflicting_groups):
            group_dir = os.path.join(
                quarantine_dir, f"conflict_{group_idx:04d}_{group['sha256'][:12]}"
            )
            for record in group["records"]:
                class_dir = os.path.join(group_dir, record["canonical_class"])
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy2(record["path"], os.path.join(class_dir, os.path.basename(record["path"])))

    def _create_report(
        self,
        records: list[dict],
        unique_records: list[dict],
        duplicate_groups: list[dict],
        conflicting_groups: list[dict],
        train_records: list[dict],
        validation_records: list[dict],
        test_records: list[dict],
    ) -> None:
        os.makedirs(self.config.cleanup_report_dir, exist_ok=True)
        report = {
            "source_image_count": len(records),
            "unique_image_count": len(unique_records),
            "duplicate_group_count": len(duplicate_groups),
            "conflicting_group_count": len(conflicting_groups),
            "quarantined_image_count": sum(len(g["records"]) for g in conflicting_groups),
            "train_image_count": len(train_records),
            "validation_image_count": len(validation_records),
            "test_image_count": len(test_records),
        }
        with open(self.config.cleanup_report_file_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=4)

    def initiate_dataset_cleanup(
        self, data_ingestion_artifact: DetectorDataIngestionArtifact
    ) -> DetectorDatasetCleanupArtifact:
        logging.info("Starting Detector Dataset Cleanup.")
        try:
            records = self._collect_images(data_ingestion_artifact.ingested_data_dir)
            hash_groups = self._group_by_hash(records)
            unique_records, duplicate_groups, conflicting_groups = self._resolve_duplicate_groups(hash_groups)

            if os.path.exists(self.config.clean_data_dir):
                shutil.rmtree(self.config.clean_data_dir)
            if os.path.exists(self.config.quarantine_dir):
                shutil.rmtree(self.config.quarantine_dir)

            train_records, validation_records, test_records = self._split_records(unique_records)

            self._copy_records(train_records, training_pipeline.CLEAN_DATASET_TRAIN_DIR_NAME, self.config.clean_data_dir)
            self._copy_records(validation_records, training_pipeline.CLEAN_DATASET_VALIDATION_DIR_NAME, self.config.clean_data_dir)
            self._copy_records(test_records, training_pipeline.CLEAN_DATASET_TEST_DIR_NAME, self.config.clean_data_dir)

            self._quarantine_conflicting_groups(conflicting_groups)
            self._create_report(
                records, unique_records, duplicate_groups, conflicting_groups,
                train_records, validation_records, test_records,
            )

            quarantined_count = sum(len(g["records"]) for g in conflicting_groups)

            artifact = DetectorDatasetCleanupArtifact(
                clean_train_data_dir=self.config.clean_train_data_dir,
                clean_validation_data_dir=self.config.clean_validation_data_dir,
                clean_test_data_dir=self.config.clean_test_data_dir,
                quarantine_dir=self.config.quarantine_dir,
                cleanup_report_file_path=self.config.cleanup_report_file_path,
                source_image_count=len(records),
                unique_image_count=len(unique_records),
                duplicate_group_count=len(duplicate_groups),
                conflicting_group_count=len(conflicting_groups),
                quarantined_image_count=quarantined_count,
            )

            logging.info(f"Detector Dataset Cleanup completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error("Error in Detector Dataset Cleanup stage.")
            raise EthiopianCoffeeException(e, sys) from e
