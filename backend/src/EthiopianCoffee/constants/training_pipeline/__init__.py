import os

# ==================================================================
# General Pipeline Constants
# ==================================================================

PIPELINE_NAME: str = "EthiopianCoffee"
ARTIFACT_DIR: str = "artifacts"
FINAL_MODEL_DIR: str = "final_model"

RANDOM_SEED: int = 42

# Image configuration
IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGE_CHANNELS: int = 3
BATCH_SIZE: int = 32

# Dataset split ratios (must sum to 1.0)
TRAIN_SPLIT_RATIO: float = 0.70
VALIDATION_SPLIT_RATIO: float = 0.15
TEST_SPLIT_RATIO: float = 0.15

# Split directory names
TRAIN_DIR_NAME: str = "train"
VALIDATION_DIR_NAME: str = "validation"
TEST_DIR_NAME: str = "test"

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
)

# Common stage artifact directory names
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_ARTIFACT_DIR: str = "ingested"

DATASET_CLEANUP_DIR_NAME: str = "dataset_cleanup"
DATASET_CLEANUP_CLEAN_DATA_DIR_NAME: str = "clean_data"
DATASET_CLEANUP_QUARANTINE_DIR_NAME: str = "quarantine"
DATASET_CLEANUP_REPORT_DIR_NAME: str = "reports"
DATASET_CLEANUP_REPORT_FILE_NAME: str = "cleanup_report.json"

CLEAN_DATASET_TRAIN_DIR_NAME: str = "train"
CLEAN_DATASET_VALIDATION_DIR_NAME: str = "validation"
CLEAN_DATASET_TEST_DIR_NAME: str = "test"

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_DIR: str = "reports"
DATA_VALIDATION_REPORT_FILE_NAME: str = "validation_report.json"

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_models"

MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_DIR_NAME: str = "reports"
MODEL_EVALUATION_CONFUSION_MATRIX_DIR_NAME: str = "confusion_matrices"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.json"

MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_FILE_NAME: str = "model.keras"
CLASS_NAMES_FILE_NAME: str = "class_names.json"
MODEL_METADATA_FILE_NAME: str = "model_metadata.json"


# ==================================================================
# Detector Pipeline Constants
# ==================================================================

DETECTOR_DATASET_DIR: str = os.path.join("datasets", "detector")
DETECTOR_ARTIFACT_DIR_NAME: str = "detector"
DETECTOR_FINAL_MODEL_SUBDIR: str = "coffee_detector"

# Canonical binary classes for detector:
# Index 0 -> not_coffee_leaf
# Index 1 -> coffee_leaf
DETECTOR_CLASS_NAMES: list[str] = [
    "not_coffee_leaf",
    "coffee_leaf",
]
DETECTOR_NUM_CLASSES: int = len(DETECTOR_CLASS_NAMES)

DETECTOR_SOURCE_CLASS_NAME_MAPPING: dict[str, str] = {
    "coffee": "coffee_leaf",
    "non_coffee": "not_coffee_leaf",
}

# Model identifiers
DETECTOR_CNN_MODEL_NAME: str = "cnn"
DETECTOR_CNN_MODEL_FILE_NAME: str = "cnn_model.keras"

DETECTOR_TRANSFER_LEARNING_MODEL_NAME: str = "efficientnetb0"
DETECTOR_TRANSFER_LEARNING_MODEL_FILE_NAME: str = "efficientnetb0_model.keras"

# Training hyperparameters
DETECTOR_EPOCHS: int = 20
DETECTOR_LEARNING_RATE: float = 0.001

DETECTOR_FINE_TUNE_EPOCHS: int = 20
DETECTOR_FINE_TUNE_LEARNING_RATE: float = 0.00001
DETECTOR_FINE_TUNE_UNFREEZE_RATIO: float = 0.30

# Decision threshold search parameters (validation dataset only)
DETECTOR_DEFAULT_DECISION_THRESHOLD: float = 0.50
DETECTOR_THRESHOLD_SEARCH_MIN: float = 0.30
DETECTOR_THRESHOLD_SEARCH_MAX: float = 0.71
DETECTOR_THRESHOLD_SEARCH_STEP: float = 0.05


# ==================================================================
# Disease Pipeline Constants
# ==================================================================

DISEASE_DATASET_DIR: str = os.path.join("datasets", "disease")
DISEASE_ARTIFACT_DIR_NAME: str = "disease"
DISEASE_FINAL_MODEL_SUBDIR: str = "coffee_disease"

# Canonical multiclass names for disease classification:
DISEASE_CLASS_NAMES: list[str] = [
    "healthy",
    "phoma",
    "leaf_rust",
    "cercospora",
]
DISEASE_NUM_CLASSES: int = len(DISEASE_CLASS_NAMES)

# Mapping from raw directory names to canonical class names
DISEASE_SOURCE_CLASS_NAME_MAPPING: dict[str, str] = {
    "Healthy": "healthy",
    "Phoma": "phoma",
    "Leaf_rust": "leaf_rust",
    "Cerscospora": "cercospora",
}

# Model identifiers
DISEASE_CNN_MODEL_NAME: str = "cnn"
DISEASE_CNN_MODEL_FILE_NAME: str = "cnn_model.keras"

DISEASE_TRANSFER_LEARNING_MODEL_NAME: str = "efficientnetb0"
DISEASE_TRANSFER_LEARNING_MODEL_FILE_NAME: str = "efficientnetb0_model.keras"

# Training hyperparameters
DISEASE_EPOCHS: int = 20
DISEASE_LEARNING_RATE: float = 0.001

DISEASE_FINE_TUNE_EPOCHS: int = 20
DISEASE_FINE_TUNE_LEARNING_RATE: float = 0.00001
DISEASE_FINE_TUNE_UNFREEZE_RATIO: float = 0.30
