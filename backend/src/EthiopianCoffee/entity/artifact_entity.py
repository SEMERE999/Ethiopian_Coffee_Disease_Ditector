from dataclasses import dataclass
from typing import Any, Optional


# ==================================================================
# Metric Artifacts
# ==================================================================

@dataclass
class ClassificationMetricArtifact:
    """
    Binary or macro-averaged classification metrics.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    true_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None


# ==================================================================
# Detector Artifacts
# ==================================================================

@dataclass
class DetectorDataIngestionArtifact:
    ingested_data_dir: str
    total_image_count: int
    class_image_counts: dict


@dataclass
class DetectorDatasetCleanupArtifact:
    clean_train_data_dir: str
    clean_validation_data_dir: str
    clean_test_data_dir: str
    quarantine_dir: str
    cleanup_report_file_path: str
    source_image_count: int
    unique_image_count: int
    duplicate_group_count: int
    conflicting_group_count: int
    quarantined_image_count: int


@dataclass
class DetectorDataValidationArtifact:
    validation_status: bool
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str
    validation_report_file_path: str
    train_image_count: int
    validation_image_count: int
    test_image_count: int


@dataclass
class DetectorDataTransformationArtifact:
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str
    image_size: tuple[int, int]
    image_channels: int
    batch_size: int
    seed: int


@dataclass
class DetectorTrainedModelArtifact:
    model_name: str
    model_file_path: str
    validation_metric_artifact: ClassificationMetricArtifact
    decision_threshold: float


@dataclass
class DetectorModelTrainerArtifact:
    trained_model_artifacts: list[DetectorTrainedModelArtifact]


@dataclass
class DetectorModelEvaluationResult:
    model_name: str
    model_file_path: str
    validation_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
    per_class_metrics: dict
    confusion_matrix_file_path: str
    decision_threshold: float
    is_accepted: bool


@dataclass
class DetectorModelEvaluationArtifact:
    evaluated_model_result: DetectorModelEvaluationResult
    selected_model_name: Optional[str]
    selected_model_file_path: Optional[str]
    selected_decision_threshold: float
    evaluation_report_file_path: str


@dataclass
class DetectorModelPusherArtifact:
    saved_model_file_path: str
    class_names_file_path: str
    model_metadata_file_path: str


# ==================================================================
# Disease Artifacts
# ==================================================================

@dataclass
class DiseaseDataIngestionArtifact:
    train_data_dir: str
    test_data_dir: str
    train_image_count: int
    test_image_count: int
    class_names: list[str]


@dataclass
class DiseaseDatasetCleanupArtifact:
    clean_train_data_dir: str
    clean_validation_data_dir: str
    clean_test_data_dir: str
    quarantine_dir: str
    cleanup_report_file_path: str
    source_image_count: int
    unique_image_count: int
    duplicate_group_count: int
    conflicting_group_count: int
    quarantined_image_count: int


@dataclass
class DiseaseDataValidationArtifact:
    validation_status: bool
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str
    validation_report_file_path: str
    train_image_count: int
    validation_image_count: int
    test_image_count: int


@dataclass
class DiseaseDataTransformationArtifact:
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str
    image_size: tuple[int, int]
    image_channels: int
    batch_size: int
    seed: int


@dataclass
class DiseaseTrainedModelArtifact:
    model_name: str
    model_file_path: str
    validation_metric_artifact: ClassificationMetricArtifact


@dataclass
class DiseaseModelTrainerArtifact:
    trained_model_artifacts: list[DiseaseTrainedModelArtifact]


@dataclass
class DiseaseModelEvaluationResult:
    model_name: str
    model_file_path: str
    test_metric_artifact: ClassificationMetricArtifact
    per_class_metrics: dict
    confusion_matrix_file_path: str
    is_accepted: bool


@dataclass
class DiseaseModelEvaluationArtifact:
    evaluated_model_artifacts: list[DiseaseModelEvaluationResult]
    selected_model_name: Optional[str]
    selected_model_file_path: Optional[str]
    evaluation_report_file_path: str


@dataclass
class DiseaseModelPusherArtifact:
    saved_model_file_path: str
    class_names_file_path: str
    model_metadata_file_path: str
