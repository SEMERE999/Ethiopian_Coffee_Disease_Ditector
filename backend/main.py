"""
Command-line entry point for the EthiopianCoffee pipeline.

Usage:
    python main.py train-detector
    python main.py train-disease
    python main.py train-all
    python main.py predict path/to/image.jpg
"""

import argparse
import sys

from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging
from EthiopianCoffee.pipeline import (
    DetectorTrainingPipeline,
    DiseaseTrainingPipeline,
    PredictionPipeline,
)
from EthiopianCoffee.entity.prediction_entity import PredictionInput


def train_detector() -> None:
    logging.info("Starting application: Detector training.")
    try:
        artifact = DetectorTrainingPipeline().run_pipeline()
        logging.info(f"Detector training completed successfully: {artifact}")
        print(f"Detector model pushed to: {artifact.saved_model_file_path}")
    except Exception as e:
        logging.error("Application failed during Detector training.")
        raise EthiopianCoffeeException(e, sys) from e


def train_disease() -> None:
    logging.info("Starting application: Disease classifier training.")
    try:
        artifact = DiseaseTrainingPipeline().run_pipeline()
        logging.info(f"Disease training completed successfully: {artifact}")
        print(f"Disease model pushed to: {artifact.saved_model_file_path}")
    except Exception as e:
        logging.error("Application failed during Disease training.")
        raise EthiopianCoffeeException(e, sys) from e


def predict(image_file_path: str) -> None:
    logging.info(f"Starting application: Prediction for {image_file_path}.")
    try:
        result = PredictionPipeline().predict(PredictionInput(image_file_path=image_file_path))

        print("Is coffee leaf:", result.is_coffee_leaf)
        print("Detector confidence:", result.detector_confidence)
        print("Detector probabilities:", result.detector_probabilities)

        if result.is_coffee_leaf:
            print("Disease:", result.predicted_class)
            print("Disease confidence:", result.confidence)
            print("Disease probabilities:", result.class_probabilities)
        else:
            print("Image rejected: this is not a coffee leaf.")

    except Exception as e:
        logging.error("Application failed during prediction.")
        raise EthiopianCoffeeException(e, sys) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="EthiopianCoffee training and prediction CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train-detector", help="Train and push the coffee-leaf detector.")
    subparsers.add_parser("train-disease", help="Train and push the disease classifier.")
    subparsers.add_parser("train-all", help="Train and push both the detector and the disease classifier.")

    predict_parser = subparsers.add_parser("predict", help="Run cascaded prediction on a single image.")
    predict_parser.add_argument("image_path", help="Path to the image file to classify.")

    args = parser.parse_args()

    if args.command == "train-detector":
        train_detector()
    elif args.command == "train-disease":
        train_disease()
    elif args.command == "train-all":
        train_detector()
        train_disease()
    elif args.command == "predict":
        predict(args.image_path)


if __name__ == "__main__":
    main()
