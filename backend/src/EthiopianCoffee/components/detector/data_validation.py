import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

from PIL import Image, UnidentifiedImageError

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DetectorDataValidationConfig
from EthiopianCoffee.entity.artifact_entity import (
    DetectorDatasetCleanupArtifact,
    DetectorDataValidationArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class DataValidation:
    """
    Data Validation component for the Coffee Leaf Detector pipeline.

    Verification checks:
    1. Train / validation / test directories exist.
    2. Both canonical classes exist in every split.
    3. Image counts in every split are non-zero.
    4. Supported file extensions only.
    5. Images can be opened (no corrupt files).
    6. No exact duplicate hashes remain within any split.
    7. No cross-split exact duplicate leakage.
    8. Directory structure is correct.
    """

    def __init__(self, data_validation_config: DetectorDataValidationConfig) -> None:
        self.config = data_validation_config
        self.expected_classes: set[str] = set(training_pipeline.DETECTOR_CLASS_NAMES)
        self.supported_extensions: set[str] = set(
            ext.lower() for ext in training_pipeline.SUPPORTED_IMAGE_EXTENSIONS
        )

    def _validate_directory_structure(self, split_dir: str, split_name: str) -> list[str]:
        errors: list[str] = []
        if not os.path.isdir(split_dir):
            errors.append(f"{split_name}: directory does not exist: {split_dir}")
            return errors

        actual_classes = {
            name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name))
        }
        missing = self.expected_classes - actual_classes
        unexpected = actual_classes - self.expected_classes

        if missing:
            errors.append(f"{split_name}: missing classes: {sorted(missing)}")
        if unexpected:
            errors.append(f"{split_name}: unexpected classes: {sorted(unexpected)}")
        return errors

    def _get_image_files(self, split_dir: str) -> dict[str, list[str]]:
        image_files: dict[str, list[str]] = {}
        for class_name in training_pipeline.DETECTOR_CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                image_files[class_name] = []
                continue

            files: list[str] = []
            for root, _, fnames in os.walk(class_dir):
                for fname in fnames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in self.supported_extensions:
                        files.append(os.path.join(root, fname))
            image_files[class_name] = sorted(files)
        return image_files

    def _calculate_sha256(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as fh:
            while chunk := fh.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _validate_images(
        self, image_files: dict[str, list[str]], split_name: str
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        corrupt_files: list[str] = []

        for class_name, file_paths in image_files.items():
            if not file_paths:
                errors.append(f"{split_name}/{class_name}: no images found.")
                continue

            for fpath in file_paths:
                if os.path.getsize(fpath) == 0:
                    errors.append(f"Zero-byte file detected: {fpath}")
                    corrupt_files.append(fpath)
                    continue

                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError, SyntaxError) as e:
                    errors.append(f"Corrupt image detected: {fpath} ({e})")
                    corrupt_files.append(fpath)

        return errors, corrupt_files

    def _check_split_duplicates(
        self, image_files: dict[str, list[str]], split_name: str
    ) -> tuple[list[str], dict[str, str]]:
        errors: list[str] = []
        split_hashes: dict[str, str] = {}

        for class_name, file_paths in image_files.items():
            for fpath in file_paths:
                h = self._calculate_sha256(fpath)
                if h in split_hashes:
                    errors.append(
                        f"{split_name}: duplicate image detected within split: "
                        f"{fpath} matches {split_hashes[h]}"
                    )
                else:
                    split_hashes[h] = fpath
        return errors, split_hashes

    def _check_cross_split_leakage(
        self, split_hashes: dict[str, dict[str, str]]
    ) -> list[str]:
        errors: list[str] = []
        split_names = list(split_hashes.keys())

        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                name_a, name_b = split_names[i], split_names[j]
                shared = set(split_hashes[name_a].keys()) & set(split_hashes[name_b].keys())
                for h in shared:
                    errors.append(
                        f"Data leakage: same image in '{name_a}' and '{name_b}': "
                        f"{split_hashes[name_a][h]} == {split_hashes[name_b][h]}"
                    )
        return errors

    def initiate_data_validation(
        self, dataset_cleanup_artifact: DetectorDatasetCleanupArtifact
    ) -> DetectorDataValidationArtifact:
        logging.info("Starting Detector Data Validation.")
        try:
            train_dir = dataset_cleanup_artifact.clean_train_data_dir
            val_dir = dataset_cleanup_artifact.clean_validation_data_dir
            test_dir = dataset_cleanup_artifact.clean_test_data_dir

            all_errors: list[str] = []
            all_corrupt: list[str] = []
            split_image_counts: dict[str, int] = {}
            split_hashes_by_split: dict[str, dict[str, str]] = {}

            for split_name, sdir in [
                ("train", train_dir),
                ("validation", val_dir),
                ("test", test_dir),
            ]:
                struct_errors = self._validate_directory_structure(sdir, split_name)
                all_errors.extend(struct_errors)

                image_files = self._get_image_files(sdir)
                total_split_imgs = sum(len(f) for f in image_files.values())
                split_image_counts[split_name] = total_split_imgs

                if total_split_imgs == 0:
                    all_errors.append(f"{split_name}: no valid images found.")

                img_errors, corrupt = self._validate_images(image_files, split_name)
                all_errors.extend(img_errors)
                all_corrupt.extend(corrupt)

                dup_errors, s_hashes = self._check_split_duplicates(image_files, split_name)
                all_errors.extend(dup_errors)
                split_hashes_by_split[split_name] = s_hashes

            leakage_errors = self._check_cross_split_leakage(split_hashes_by_split)
            all_errors.extend(leakage_errors)

            validation_status = len(all_errors) == 0

            os.makedirs(self.config.validation_report_dir, exist_ok=True)
            report = {
                "validation_status": validation_status,
                "split_image_counts": split_image_counts,
                "error_count": len(all_errors),
                "corrupt_file_count": len(all_corrupt),
                "errors": all_errors,
            }
            with open(self.config.validation_report_file_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=4)

            artifact = DetectorDataValidationArtifact(
                validation_status=validation_status,
                train_data_dir=train_dir,
                validation_data_dir=val_dir,
                test_data_dir=test_dir,
                validation_report_file_path=self.config.validation_report_file_path,
                train_image_count=split_image_counts.get("train", 0),
                validation_image_count=split_image_counts.get("validation", 0),
                test_image_count=split_image_counts.get("test", 0),
            )

            logging.info(f"Detector Data Validation completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error("Error in Detector Data Validation stage.")
            raise EthiopianCoffeeException(e, sys) from e
