import json
import os
import sys

import numpy as np
import tensorflow as tf

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging
from EthiopianCoffee.entity.prediction_entity import PredictionInput, PredictionOutput


class PredictionPipeline:
    """
    Cascaded inference pipeline: Detector -> Disease Classifier.

    Stage 1 (gate): the Detector decides whether the image is a coffee
    leaf at all, using the decision threshold tuned during its own
    Model Evaluation stage (read back from model_metadata.json, never
    re-guessed here).

    Stage 2 (classify): only runs if the Detector accepts the image.
    The Disease Classifier then predicts which of the 4 disease
    classes the leaf belongs to.

    Both final models are loaded once, lazily, on first use, and
    reused across calls to `predict`.
    """

    def __init__(self) -> None:
        self._detector_model: tf.keras.Model | None = None
        self._detector_class_names: list[str] | None = None
        self._detector_decision_threshold: float | None = None

        self._disease_model: tf.keras.Model | None = None
        self._disease_class_names: list[str] | None = None

        self._image_size = training_pipeline.IMAGE_SIZE

    # ========================================================
    # MODEL / METADATA LOADING
    # ========================================================

    @staticmethod
    def _final_model_dir(subdir: str) -> str:
        return os.path.join(training_pipeline.FINAL_MODEL_DIR, subdir)

    @staticmethod
    def _load_class_names(class_names_file_path: str) -> list[str]:
        if not os.path.isfile(class_names_file_path):
            raise FileNotFoundError(f"Class names file not found: {class_names_file_path}")

        with open(class_names_file_path, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)

        # mapping is {"0": "class_a", "1": "class_b", ...}
        return [mapping[str(idx)] for idx in range(len(mapping))]

    def _load_detector(self) -> None:
        if self._detector_model is not None:
            return

        model_dir = self._final_model_dir(training_pipeline.DETECTOR_FINAL_MODEL_SUBDIR)
        model_path = os.path.join(model_dir, training_pipeline.MODEL_FILE_NAME)
        class_names_path = os.path.join(model_dir, training_pipeline.CLASS_NAMES_FILE_NAME)
        metadata_path = os.path.join(model_dir, training_pipeline.MODEL_METADATA_FILE_NAME)

        logging.info(f"Loading detector model from: {model_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")

        self._detector_model = tf.keras.models.load_model(model_path)
        self._detector_class_names = self._load_class_names(class_names_path)

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Detector model metadata not found: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        self._detector_decision_threshold = float(
            metadata.get("decision_threshold", training_pipeline.DETECTOR_DEFAULT_DECISION_THRESHOLD)
        )
        logging.info(f"Detector decision threshold: {self._detector_decision_threshold}")

    def _load_disease_classifier(self) -> None:
        if self._disease_model is not None:
            return

        model_dir = self._final_model_dir(training_pipeline.DISEASE_FINAL_MODEL_SUBDIR)
        model_path = os.path.join(model_dir, training_pipeline.MODEL_FILE_NAME)
        class_names_path = os.path.join(model_dir, training_pipeline.CLASS_NAMES_FILE_NAME)

        logging.info(f"Loading disease classifier model from: {model_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Disease classifier model not found: {model_path}")

        self._disease_model = tf.keras.models.load_model(model_path)
        self._disease_class_names = self._load_class_names(class_names_path)

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    def _load_image_batch(self, image_file_path: str) -> np.ndarray:
        """
        Load and resize a single image into a (1, H, W, C) float32 batch.

        Raw [0, 255] pixel values are kept as-is: both the detector and
        disease models include their own internal Rescaling(1/255) layer,
        so normalizing here again would silently halve the intended input.
        """
        if not os.path.isfile(image_file_path):
            raise FileNotFoundError(f"Image file not found: {image_file_path}")

        image = tf.keras.utils.load_img(
            image_file_path,
            target_size=self._image_size,
            color_mode="rgb",
        )
        image_array = tf.keras.utils.img_to_array(image)
        return np.expand_dims(image_array, axis=0)

    # ========================================================
    # MAIN PREDICTION METHOD
    # ========================================================

    def predict(self, prediction_input: PredictionInput) -> PredictionOutput:
        logging.info(f"Starting cascaded prediction for: {prediction_input.image_file_path}")
        try:
            self._load_detector()

            image_batch = self._load_image_batch(prediction_input.image_file_path)

            # ------------------------------------------------
            # Stage 1: Detector gate
            # ------------------------------------------------
            detector_prob = float(self._detector_model.predict(image_batch, verbose=0)[0][0])
            not_leaf_name, leaf_name = self._detector_class_names
            detector_probabilities = {
                not_leaf_name: 1.0 - detector_prob,
                leaf_name: detector_prob,
            }
            is_coffee_leaf = detector_prob >= self._detector_decision_threshold
            detector_confidence = detector_prob if is_coffee_leaf else (1.0 - detector_prob)

            logging.info(
                f"Detector result: is_coffee_leaf={is_coffee_leaf}, "
                f"confidence={detector_confidence:.4f}"
            )

            if not is_coffee_leaf:
                return PredictionOutput(
                    is_coffee_leaf=False,
                    detector_confidence=detector_confidence,
                    detector_probabilities=detector_probabilities,
                    predicted_class=None,
                    confidence=None,
                    class_probabilities=None,
                )

            # ------------------------------------------------
            # Stage 2: Disease classification
            # ------------------------------------------------
            self._load_disease_classifier()

            disease_probs = self._disease_model.predict(image_batch, verbose=0)[0]
            predicted_index = int(np.argmax(disease_probs))
            predicted_class = self._disease_class_names[predicted_index]
            confidence = float(disease_probs[predicted_index])
            class_probabilities = {
                class_name: float(prob)
                for class_name, prob in zip(self._disease_class_names, disease_probs)
            }

            logging.info(f"Disease result: predicted_class={predicted_class}, confidence={confidence:.4f}")

            return PredictionOutput(
                is_coffee_leaf=True,
                detector_confidence=detector_confidence,
                detector_probabilities=detector_probabilities,
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probabilities,
            )

        except Exception as e:
            logging.error("Error occurred during cascaded prediction.")
            raise EthiopianCoffeeException(e, sys) from e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cascaded coffee-leaf prediction on a single image.")
    parser.add_argument("image_path", help="Path to the image file to classify.")
    args = parser.parse_args()

    result = PredictionPipeline().predict(PredictionInput(image_file_path=args.image_path))

    print("Is coffee leaf:", result.is_coffee_leaf)
    print("Detector confidence:", result.detector_confidence)
    print("Detector probabilities:", result.detector_probabilities)

    if result.is_coffee_leaf:
        print("Disease:", result.predicted_class)
        print("Disease confidence:", result.confidence)
        print("Disease probabilities:", result.class_probabilities)
    else:
        print("Image rejected: this is not a coffee leaf.")
