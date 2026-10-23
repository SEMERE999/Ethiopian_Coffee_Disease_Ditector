import json
import os
import shutil
import sys
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DetectorModelPusherConfig
from EthiopianCoffee.entity.artifact_entity import (
    DetectorModelEvaluationArtifact,
    DetectorModelPusherArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class ModelPusher:
    """
    Model Pusher component for the Coffee Leaf Detector pipeline.

    Pushes the selected model to final_model/coffee_detector/:
      1. Validates model loading and output shape (None, 1).
      2. Validates decision threshold in (0, 1).
      3. Copies model to final_model/coffee_detector/model.keras.
      4. Saves class_names.json.
      5. Saves model_metadata.json with tuned decision threshold.
      6. Verifies pushed model loads successfully.
    """

    def __init__(self, model_pusher_config: DetectorModelPusherConfig) -> None:
        self.config = model_pusher_config

    def _validate_before_push(self, model_file_path: str, decision_threshold: float) -> tf.keras.Model:
        logging.info(f"Validating detector model before push: {model_file_path}")
        if not os.path.isfile(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        try:
            model = tf.keras.models.load_model(model_file_path)
        except Exception as e:
            raise ValueError(f"Model could not be loaded: {model_file_path}") from e

        output_shape = model.output_shape
        if output_shape[-1] != 1:
            raise ValueError(
                f"Model output dimension must be 1 (binary sigmoid). Got: {output_shape}"
            )

        if not (0.0 < decision_threshold < 1.0):
            raise ValueError(
                f"Decision threshold must be in (0, 1). Got: {decision_threshold}"
            )

        logging.info(f"Pre-push validation passed. Output: {output_shape}, threshold: {decision_threshold}")
        return model

    def _push_model(self, source_path: str) -> str:
        dest_path = self.config.saved_model_file_path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)
        logging.info(f"Detector model pushed to: {dest_path}")
        return dest_path

    def _save_class_names(self) -> str:
        class_names_path = self.config.class_names_file_path
        os.makedirs(os.path.dirname(class_names_path), exist_ok=True)

        class_mapping = {
            str(idx): name
            for idx, name in enumerate(training_pipeline.DETECTOR_CLASS_NAMES)
        }
        with open(class_names_path, "w", encoding="utf-8") as fh:
            json.dump(class_mapping, fh, indent=4)

        logging.info(f"Detector class names saved: {class_names_path}")
        return class_names_path

    def _save_model_metadata(
        self,
        model_name: str,
        decision_threshold: float,
        model_evaluation_artifact: DetectorModelEvaluationArtifact,
    ) -> str:
        metadata_path = self.config.model_metadata_file_path
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        test_metrics = model_evaluation_artifact.evaluated_model_result.test_metric_artifact

        metadata = {
            "model_name": model_name,
            "model_file_name": training_pipeline.MODEL_FILE_NAME,
            "class_names": training_pipeline.DETECTOR_CLASS_NAMES,
            "num_classes": training_pipeline.DETECTOR_NUM_CLASSES,
            "image_size": list(training_pipeline.IMAGE_SIZE),
            "image_channels": training_pipeline.IMAGE_CHANNELS,
            "decision_threshold": decision_threshold,
            "evaluation_metrics": {
                "accuracy": test_metrics.accuracy,
                "precision": test_metrics.precision,
                "recall": test_metrics.recall,
                "f1_score": test_metrics.f1_score,
                "auc": test_metrics.auc,
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4)

        logging.info(f"Detector model metadata saved: {metadata_path}")
        return metadata_path

    def _verify_pushed_model(self, pushed_model_path: str) -> None:
        logging.info("Verifying pushed detector model.")
        if not os.path.isfile(pushed_model_path):
            raise FileNotFoundError(f"Pushed model not found: {pushed_model_path}")

        model = tf.keras.models.load_model(pushed_model_path)
        if model.output_shape[-1] != 1:
            raise ValueError(
                f"Pushed model output dimension mismatch. Expected 1, got {model.output_shape[-1]}"
            )
        logging.info("Pushed detector model verified successfully.")

    def initiate_model_pusher(
        self, model_evaluation_artifact: DetectorModelEvaluationArtifact
    ) -> DetectorModelPusherArtifact:
        logging.info("Starting Detector Model Pusher.")
        try:
            selected_model_file_path = model_evaluation_artifact.selected_model_file_path
            decision_threshold = model_evaluation_artifact.selected_decision_threshold
            selected_model_name = model_evaluation_artifact.selected_model_name

            if selected_model_file_path is None or not os.path.isfile(selected_model_file_path):
                raise ValueError("No valid model was selected to be pushed.")

            self._validate_before_push(selected_model_file_path, decision_threshold)
            dest_model_path = self._push_model(selected_model_file_path)
            class_names_path = self._save_class_names()
            metadata_path = self._save_model_metadata(
                selected_model_name, decision_threshold, model_evaluation_artifact
            )
            self._verify_pushed_model(dest_model_path)

            artifact = DetectorModelPusherArtifact(
                saved_model_file_path=dest_model_path,
                class_names_file_path=class_names_path,
                model_metadata_file_path=metadata_path,
            )

            logging.info(f"Detector Model Pusher completed: {artifact}")
            return artifact

        except Exception as e:
            logging.error("Error in Detector Model Pusher stage.")
            raise EthiopianCoffeeException(e, sys) from e
