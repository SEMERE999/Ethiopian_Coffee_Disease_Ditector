import sys

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging

from EthiopianCoffee.entity.config_entity import (
    TrainingPipelineConfig,
    DetectorDataIngestionConfig,
    DetectorDatasetCleanupConfig,
    DetectorDataValidationConfig,
    DetectorDataTransformationConfig,
    DetectorModelTrainerConfig,
    DetectorModelEvaluationConfig,
    DetectorModelPusherConfig,
)
from EthiopianCoffee.entity.artifact_entity import DetectorModelPusherArtifact

from EthiopianCoffee.components.detector import (
    DataIngestion,
    DatasetCleanup,
    DataValidation,
    DataTransformation,
    ModelTrainer,
    ModelEvaluation,
    ModelPusher,
)


class DetectorTrainingPipeline:
    """
    End-to-end training pipeline for the coffee-leaf Detector (binary).

    Stages: Data Ingestion -> Dataset Cleanup -> Data Validation ->
    Data Transformation -> Model Trainer -> Model Evaluation -> Model Pusher.

    Each run gets its own timestamped artifact directory under
    artifacts/detector/<timestamp>/, kept separate from the disease
    pipeline's artifacts.
    """

    def __init__(self) -> None:
        self.training_pipeline_config = TrainingPipelineConfig(
            pipeline_subdir=training_pipeline.DETECTOR_ARTIFACT_DIR_NAME
        )

    def run_pipeline(self) -> DetectorModelPusherArtifact:
        logging.info("=" * 80)
        logging.info("Starting Coffee Leaf Detector Training Pipeline")
        logging.info("=" * 80)

        try:
            # 1. DATA INGESTION
            data_ingestion_config = DetectorDataIngestionConfig(self.training_pipeline_config)
            data_ingestion_artifact = DataIngestion(data_ingestion_config).initiate_data_ingestion()
            logging.info(f"Detector Data Ingestion Artifact: {data_ingestion_artifact}")

            # 2. DATASET CLEANUP
            dataset_cleanup_config = DetectorDatasetCleanupConfig(self.training_pipeline_config)
            dataset_cleanup_artifact = DatasetCleanup(dataset_cleanup_config).initiate_dataset_cleanup(
                data_ingestion_artifact
            )
            logging.info(f"Detector Dataset Cleanup Artifact: {dataset_cleanup_artifact}")

            # 3. DATA VALIDATION
            data_validation_config = DetectorDataValidationConfig(self.training_pipeline_config)
            data_validation_artifact = DataValidation(data_validation_config).initiate_data_validation(
                dataset_cleanup_artifact
            )
            logging.info(f"Detector Data Validation Artifact: {data_validation_artifact}")

            if not data_validation_artifact.validation_status:
                raise ValueError("Detector data validation failed. Pipeline execution stopped.")

            # 4. DATA TRANSFORMATION
            data_transformation_config = DetectorDataTransformationConfig(self.training_pipeline_config)
            data_transformation_artifact = DataTransformation(
                data_transformation_config
            ).initiate_data_transformation(data_validation_artifact)
            logging.info(f"Detector Data Transformation Artifact: {data_transformation_artifact}")

            # 5. MODEL TRAINING
            model_trainer_config = DetectorModelTrainerConfig(self.training_pipeline_config)
            model_trainer_artifact = ModelTrainer(model_trainer_config).initiate_model_training(
                data_transformation_artifact
            )
            logging.info(f"Detector Model Trainer Artifact: {model_trainer_artifact}")

            # 6. MODEL EVALUATION
            model_evaluation_config = DetectorModelEvaluationConfig(self.training_pipeline_config)
            model_evaluation_artifact = ModelEvaluation(model_evaluation_config).initiate_model_evaluation(
                model_trainer_artifact, data_transformation_artifact
            )
            logging.info(f"Detector Model Evaluation Artifact: {model_evaluation_artifact}")

            # 7. MODEL PUSHING
            model_pusher_config = DetectorModelPusherConfig(self.training_pipeline_config)
            model_pusher_artifact = ModelPusher(model_pusher_config).initiate_model_pusher(
                model_evaluation_artifact
            )
            logging.info(f"Detector Model Pusher Artifact: {model_pusher_artifact}")

            logging.info("=" * 80)
            logging.info("Coffee Leaf Detector Training Pipeline completed successfully.")
            logging.info("=" * 80)

            return model_pusher_artifact

        except Exception as e:
            logging.error("Error occurred during Detector Training Pipeline execution.")
            raise EthiopianCoffeeException(e, sys) from e


if __name__ == "__main__":
    DetectorTrainingPipeline().run_pipeline()
