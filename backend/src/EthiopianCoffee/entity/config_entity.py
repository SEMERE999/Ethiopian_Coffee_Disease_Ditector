import os
from datetime import datetime

from EthiopianCoffee.constants import training_pipeline


# ==================================================================
# Base Training Pipeline Config
# ==================================================================

class TrainingPipelineConfig:
    """
    Common configuration for a single pipeline execution run.
    Creates timestamped artifact directories.
    """

    def __init__(self, timestamp: datetime = None, pipeline_subdir: str = "") -> None:
        if timestamp is None:
            timestamp = datetime.now()

        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.timestamp: str = timestamp_str
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_root: str = training_pipeline.ARTIFACT_DIR

        # artifacts/<pipeline_subdir>/<timestamp>
        if pipeline_subdir:
            self.artifact_dir: str = os.path.join(
                self.artifact_root,
                pipeline_subdir,
                timestamp_str,
            )
        else:
            self.artifact_dir: str = os.path.join(
                self.artifact_root,
                timestamp_str,
            )

        self.final_model_root: str = training_pipeline.FINAL_MODEL_DIR


# ==================================================================
# Detector Configuration Entities
# ==================================================================

class DetectorDataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.dataset_dir: str = training_pipeline.DETECTOR_DATASET_DIR
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )
        self.ingested_data_dir: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_ARTIFACT_DIR,
        )


class DetectorDatasetCleanupConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_cleanup_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATASET_CLEANUP_DIR_NAME,
        )
        self.clean_data_dir: str = os.path.join(
            self.data_cleanup_dir,
            training_pipeline.DATASET_CLEANUP_CLEAN_DATA_DIR_NAME,
        )
        self.clean_train_data_dir: str = os.path.join(
            self.clean_data_dir,
            training_pipeline.CLEAN_DATASET_TRAIN_DIR_NAME,
        )
        self.clean_validation_data_dir: str = os.path.join(
            self.clean_data_dir,
            training_pipeline.CLEAN_DATASET_VALIDATION_DIR_NAME,
        )
        self.clean_test_data_dir: str = os.path.join(
            self.clean_data_dir,
            training_pipeline.CLEAN_DATASET_TEST_DIR_NAME,
        )
        self.quarantine_dir: str = os.path.join(
            self.data_cleanup_dir,
            training_pipeline.DATASET_CLEANUP_QUARANTINE_DIR_NAME,
        )
        self.cleanup_report_dir: str = os.path.join(
            self.data_cleanup_dir,
            training_pipeline.DATASET_CLEANUP_REPORT_DIR_NAME,
        )
        self.cleanup_report_file_path: str = os.path.join(
            self.cleanup_report_dir,
            training_pipeline.DATASET_CLEANUP_REPORT_FILE_NAME,
        )
        self.train_split_ratio: float = training_pipeline.TRAIN_SPLIT_RATIO
        self.validation_split_ratio: float = training_pipeline.VALIDATION_SPLIT_RATIO
        self.test_split_ratio: float = training_pipeline.TEST_SPLIT_RATIO
        self.random_seed: int = training_pipeline.RANDOM_SEED


class DetectorDataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )
        self.validation_report_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT_DIR,
        )
        self.validation_report_file_path: str = os.path.join(
            self.validation_report_dir,
            training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME,
        )


class DetectorDataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE
        self.image_channels: int = training_pipeline.IMAGE_CHANNELS
        self.batch_size: int = training_pipeline.BATCH_SIZE
        self.seed: int = training_pipeline.RANDOM_SEED


class DetectorModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_dir: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
        )
        self.cnn_model_file_path: str = os.path.join(
            self.trained_model_dir,
            training_pipeline.DETECTOR_CNN_MODEL_FILE_NAME,
        )
        self.transfer_learning_model_file_path: str = os.path.join(
            self.trained_model_dir,
            training_pipeline.DETECTOR_TRANSFER_LEARNING_MODEL_FILE_NAME,
        )
        self.batch_size: int = training_pipeline.BATCH_SIZE
        self.epochs: int = training_pipeline.DETECTOR_EPOCHS
        self.learning_rate: float = training_pipeline.DETECTOR_LEARNING_RATE
        self.fine_tune_epochs: int = training_pipeline.DETECTOR_FINE_TUNE_EPOCHS
        self.fine_tune_learning_rate: float = training_pipeline.DETECTOR_FINE_TUNE_LEARNING_RATE
        self.fine_tune_unfreeze_ratio: float = training_pipeline.DETECTOR_FINE_TUNE_UNFREEZE_RATIO
        self.image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE
        self.image_channels: int = training_pipeline.IMAGE_CHANNELS
        self.num_classes: int = training_pipeline.DETECTOR_NUM_CLASSES
        self.seed: int = training_pipeline.RANDOM_SEED


class DetectorModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_EVALUATION_DIR_NAME,
        )
        self.evaluation_report_dir: str = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_DIR_NAME,
        )
        self.confusion_matrix_dir: str = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_CONFUSION_MATRIX_DIR_NAME,
        )
        self.evaluation_report_file_path: str = os.path.join(
            self.evaluation_report_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_FILE_NAME,
        )


class DetectorModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_pusher_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_PUSHER_DIR_NAME,
        )
        self.saved_model_dir: str = os.path.join(
            training_pipeline_config.final_model_root,
            training_pipeline.DETECTOR_FINAL_MODEL_SUBDIR,
        )
        self.saved_model_file_path: str = os.path.join(
            self.saved_model_dir,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.class_names_file_path: str = os.path.join(
            self.saved_model_dir,
            training_pipeline.CLASS_NAMES_FILE_NAME,
        )
        self.model_metadata_file_path: str = os.path.join(
            self.saved_model_dir,
            training_pipeline.MODEL_METADATA_FILE_NAME,
        )


# ==================================================================
# Disease Configuration Entities
# ==================================================================

class DiseaseDataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.dataset_dir: str = training_pipeline.DISEASE_DATASET_DIR
        self.train_source_dir: str = os.path.join(
            self.dataset_dir,
            training_pipeline.TRAIN_DIR_NAME,
        )
        self.test_source_dir: str = os.path.join(
            self.dataset_dir,
            training_pipeline.TEST_DIR_NAME,
        )
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )
        self.ingested_data_dir: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_ARTIFACT_DIR,
        )


class DiseaseDatasetCleanupConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_cleanup_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATASET_CLEANUP_DIR_NAME,
        )
        self.clean_data_dir: str = os.path.join(
            self.data_cleanup_dir,
            training_pipeline.DATASET_CLEANUP_CLEAN_DATA_DIR_NAME,
        )
        self.clean_train_data_dir: str = os.path.join(
            self.clean_data_dir,
            training_pipeline.CLEAN_DATASET_TRAIN_DIR_NAME,
        )
        self.clean_validation_data_dir: str = os.path.join(
            self.clean_data_dir,
            training_pipeline.CLEAN_DATASET_VALIDATION_DIR_NAME,
        )
        self.clean_test_data_dir: str = os.path.join(
            self.clean_data_dir,
            training_pipeline.CLEAN_DATASET_TEST_DIR_NAME,
        )
        self.quarantine_dir: str = os.path.join(
            self.data_cleanup_dir,
            training_pipeline.DATASET_CLEANUP_QUARANTINE_DIR_NAME,
        )
        self.cleanup_report_dir: str = os.path.join(
            self.data_cleanup_dir,
            training_pipeline.DATASET_CLEANUP_REPORT_DIR_NAME,
        )
        self.cleanup_report_file_path: str = os.path.join(
            self.cleanup_report_dir,
            training_pipeline.DATASET_CLEANUP_REPORT_FILE_NAME,
        )
        self.train_split_ratio: float = training_pipeline.TRAIN_SPLIT_RATIO
        self.validation_split_ratio: float = training_pipeline.VALIDATION_SPLIT_RATIO
        self.test_split_ratio: float = training_pipeline.TEST_SPLIT_RATIO
        self.random_seed: int = training_pipeline.RANDOM_SEED


class DiseaseDataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )
        self.validation_report_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT_DIR,
        )
        self.validation_report_file_path: str = os.path.join(
            self.validation_report_dir,
            training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME,
        )


class DiseaseDataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE
        self.image_channels: int = training_pipeline.IMAGE_CHANNELS
        self.batch_size: int = training_pipeline.BATCH_SIZE
        self.seed: int = training_pipeline.RANDOM_SEED


class DiseaseModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_dir: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
        )
        self.cnn_model_file_path: str = os.path.join(
            self.trained_model_dir,
            training_pipeline.DISEASE_CNN_MODEL_FILE_NAME,
        )
        self.transfer_learning_model_file_path: str = os.path.join(
            self.trained_model_dir,
            training_pipeline.DISEASE_TRANSFER_LEARNING_MODEL_FILE_NAME,
        )
        self.batch_size: int = training_pipeline.BATCH_SIZE
        self.epochs: int = training_pipeline.DISEASE_EPOCHS
        self.learning_rate: float = training_pipeline.DISEASE_LEARNING_RATE
        self.fine_tune_epochs: int = training_pipeline.DISEASE_FINE_TUNE_EPOCHS
        self.fine_tune_learning_rate: float = training_pipeline.DISEASE_FINE_TUNE_LEARNING_RATE
        self.fine_tune_unfreeze_ratio: float = training_pipeline.DISEASE_FINE_TUNE_UNFREEZE_RATIO
        self.image_size: tuple[int, int] = training_pipeline.IMAGE_SIZE
        self.image_channels: int = training_pipeline.IMAGE_CHANNELS
        self.num_classes: int = training_pipeline.DISEASE_NUM_CLASSES
        self.seed: int = training_pipeline.RANDOM_SEED


class DiseaseModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_EVALUATION_DIR_NAME,
        )
        self.evaluation_report_dir: str = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_DIR_NAME,
        )
        self.confusion_matrix_dir: str = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_CONFUSION_MATRIX_DIR_NAME,
        )
        self.evaluation_report_file_path: str = os.path.join(
            self.evaluation_report_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_FILE_NAME,
        )


class DiseaseModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_pusher_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_PUSHER_DIR_NAME,
        )
        self.saved_model_dir: str = os.path.join(
            training_pipeline_config.final_model_root,
            training_pipeline.DISEASE_FINAL_MODEL_SUBDIR,
        )
        self.saved_model_file_path: str = os.path.join(
            self.saved_model_dir,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.class_names_file_path: str = os.path.join(
            self.saved_model_dir,
            training_pipeline.CLASS_NAMES_FILE_NAME,
        )
        self.model_metadata_file_path: str = os.path.join(
            self.saved_model_dir,
            training_pipeline.MODEL_METADATA_FILE_NAME,
        )
