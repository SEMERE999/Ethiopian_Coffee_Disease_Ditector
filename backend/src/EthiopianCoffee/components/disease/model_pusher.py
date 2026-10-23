import json
import os
import shutil
import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DiseaseModelPusherConfig
from EthiopianCoffee.entity.artifact_entity import (
    DiseaseModelEvaluationArtifact,
    DiseaseModelPusherArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class ModelPusher:
    """
    Model Pusher component for the Coffee Leaf Disease Classification pipeline.

    Pushes the selected model to final_model/coffee_disease/:
      1. Validates model loading and output shape (None, num_classes).
      2. Copies model to final_model/coffee_disease/model.keras.
      3. Saves class_names.json.
      4. Saves model_metadata.json with evaluation metrics.
      5. Verifies the pushed model, class names, and metadata all round-trip
         correctly (catches silent corruption on copy/serialize, not just
         missing files).
    """

    def __init__(self, model_pusher_config: DiseaseModelPusherConfig) -> None:
        self.config = model_pusher_config

    def _validate_before_push(self, model_file_path: str) -> tf.keras.Model:
        logging.info(f"Validating disease model before push: {model_file_path}")
        if not os.path.isfile(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        try:
            model = tf.keras.models.load_model(model_file_path)
        except Exception as e:
            raise ValueError(f"Model could not be loaded: {model_file_path}") from e

        output_shape = model.output_shape
        expected_units = training_pipeline.DISEASE_NUM_CLASSES
        if output_shape[-1] != expected_units:
            raise ValueError(
                f"Model output dimension must be {expected_units} (softmax multiclass). "
                f"Got: {output_shape}"
            )

        logging.info(f"Pre-push validation passed. Output: {output_shape}")
        return model

    def _push_model(self, source_path: str) -> str:
        dest_path = self.config.saved_model_file_path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)
        logging.info(f"Disease model pushed to: {dest_path}")
        return dest_path

    def _save_class_names(self) -> str:
        class_names_path = self.config.class_names_file_path
        os.makedirs(os.path.dirname(class_names_path), exist_ok=True)

        class_mapping = {
            str(idx): name
            for idx, name in enumerate(training_pipeline.DISEASE_CLASS_NAMES)
        }
        with open(class_names_path, "w", encoding="utf-8") as fh:
            json.dump(class_mapping, fh, indent=4)

        logging.info(f"Disease class names saved: {class_names_path}")
        return class_names_path

    def _save_model_metadata(
        self,
        model_name: str,
        model_evaluation_artifact: DiseaseModelEvaluationArtifact,
    ) -> str:
        metadata_path = self.config.model_metadata_file_path
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        selected_result = next(
            (
                result
                for result in model_evaluation_artifact.evaluated_model_artifacts
                if result.model_name == model_name
            ),
            model_evaluation_artifact.evaluated_model_artifacts[0],
        )
        test_metrics = selected_result.test_metric_artifact

        metadata = {
            "model_name": model_name,
            "model_file_name": training_pipeline.MODEL_FILE_NAME,
            "class_names": training_pipeline.DISEASE_CLASS_NAMES,
            "num_classes": training_pipeline.DISEASE_NUM_CLASSES,
            "image_size": list(training_pipeline.IMAGE_SIZE),
            "image_channels": training_pipeline.IMAGE_CHANNELS,
            "evaluation_metrics": {
                "accuracy": test_metrics.accuracy,
                "macro_precision": test_metrics.precision,
                "macro_recall": test_metrics.recall,
                "macro_f1": test_metrics.f1_score,
            },
            "per_class_metrics": selected_result.per_class_metrics,
        }

        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4)

        logging.info(f"Disease model metadata saved: {metadata_path}")
        return metadata_path

    def _verify_pushed_model(self, pushed_model_path: str) -> None:
        logging.info("Verifying pushed disease model.")
        if not os.path.isfile(pushed_model_path):
            raise FileNotFoundError(f"Pushed model not found: {pushed_model_path}")

        model = tf.keras.models.load_model(pushed_model_path)
        expected_units = training_pipeline.DISEASE_NUM_CLASSES
        if model.output_shape[-1] != expected_units:
            raise ValueError(
                f"Pushed model output dimension mismatch. "
                f"Expected {expected_units}, got {model.output_shape[-1]}"
            )
        logging.info("Pushed disease model verified successfully.")

    def _verify_class_names(self, class_names_file_path: str) -> None:
        with open(class_names_file_path, "r", encoding="utf-8") as fh:
            class_mapping = json.load(fh)

        expected_mapping = {
            str(idx): name
            for idx, name in enumerate(training_pipeline.DISEASE_CLASS_NAMES)
        }
        if class_mapping != expected_mapping:
            raise ValueError("Saved disease class mapping does not match the expected mapping.")

        logging.info("Disease class-name mapping verified successfully.")

    def _verify_model_metadata(self, model_metadata_file_path: str) -> None:
        with open(model_metadata_file_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        if metadata["num_classes"] != training_pipeline.DISEASE_NUM_CLASSES:
            raise ValueError("Disease model metadata contains an incorrect number of classes.")
        if metadata["class_names"] != training_pipeline.DISEASE_CLASS_NAMES:
            raise ValueError("Disease model metadata contains an incorrect class-name list.")
        if tuple(metadata["image_size"]) != training_pipeline.IMAGE_SIZE:
            raise ValueError("Disease model metadata contains an incorrect image size.")

        logging.info("Disease model metadata verified successfully.")

    def initiate_model_pusher(
        self, model_evaluation_artifact: DiseaseModelEvaluationArtifact
    ) -> DiseaseModelPusherArtifact:
        logging.info("Starting Disease Model Pusher.")
        try:
            selected_model_file_path = model_evaluation_artifact.selected_model_file_path
            selected_model_name = model_evaluation_artifact.selected_model_name

            if selected_model_name is None or selected_model_file_path is None:
                raise ValueError("No valid model was selected to be pushed.")
            if not os.path.isfile(selected_model_file_path):
                raise ValueError(f"Selected model file does not exist: {selected_model_file_path}")

            self._validate_before_push(selected_model_file_path)
            dest_model_path = self._push_model(selected_model_file_path)
            class_names_path = self._save_class_names()
            metadata_path = self._save_model_metadata(
                selected_model_name, model_evaluation_artifact
            )

            self._verify_pushed_model(dest_model_path)
            self._verify_class_names(class_names_path)
            self._verify_model_metadata(metadata_path)

            artifact = DiseaseModelPusherArtifact(
                saved_model_file_path=dest_model_path,
                class_names_file_path=class_names_path,
                model_metadata_file_path=metadata_path,
            )

            logging.info(f"Disease Model Pusher completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error("Error in Disease Model Pusher stage.")
            raise EthiopianCoffeeException(e, sys) from e
