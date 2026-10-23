import json
import os
import sys
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DiseaseModelEvaluationConfig
from EthiopianCoffee.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DiseaseDataTransformationArtifact,
    DiseaseModelEvaluationArtifact,
    DiseaseModelEvaluationResult,
    DiseaseModelTrainerArtifact,
    DiseaseTrainedModelArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class ModelEvaluation:
    """
    Evaluates trained multiclass coffee-leaf disease classifier models.

    Evaluation flow:
      1. Create validation and test datasets.
      2. For each trained model, generate validation predictions and
         macro-averaged metrics (independently recomputed here rather
         than trusting the trainer's self-reported numbers).
      3. Select the best model using validation macro-F1 (accuracy as
         tie-breaker). The test set is NOT used for selection.
      4. Evaluate the selected model ONCE on the untouched test set.
      5. Save per-class metrics, the confusion matrix, and the report.
    """

    def __init__(self, model_evaluation_config: DiseaseModelEvaluationConfig) -> None:
        self.config = model_evaluation_config

    def _create_dataset(
        self, data_dir: str, data_transformation_artifact: DiseaseDataTransformationArtifact
    ) -> tf.data.Dataset:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="int",
            class_names=training_pipeline.DISEASE_CLASS_NAMES,
            image_size=data_transformation_artifact.image_size,
            batch_size=data_transformation_artifact.batch_size,
            shuffle=False,
            color_mode="rgb",
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def _get_predictions(
        self, model: tf.keras.Model, dataset: tf.data.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        true_labels: list[int] = []
        pred_labels: list[int] = []

        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            pred_labels.extend(np.argmax(preds, axis=1).tolist())
            true_labels.extend(labels.numpy().tolist())

        return (
            np.array(true_labels, dtype=np.int32),
            np.array(pred_labels, dtype=np.int32),
        )

    def _calculate_metrics(
        self, true_np: np.ndarray, pred_np: np.ndarray
    ) -> ClassificationMetricArtifact:
        acc = float(accuracy_score(true_np, pred_np))
        prec = float(precision_score(true_np, pred_np, average="macro", zero_division=0))
        rec = float(recall_score(true_np, pred_np, average="macro", zero_division=0))
        f1 = float(f1_score(true_np, pred_np, average="macro", zero_division=0))

        return ClassificationMetricArtifact(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
        )

    def _calculate_per_class_metrics(
        self, true_np: np.ndarray, pred_np: np.ndarray
    ) -> dict[str, Any]:
        class_names = training_pipeline.DISEASE_CLASS_NAMES
        report = classification_report(
            true_np,
            pred_np,
            labels=list(range(training_pipeline.DISEASE_NUM_CLASSES)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        per_class_metrics: dict[str, Any] = {}
        for class_name in class_names:
            class_metrics = report[class_name]
            per_class_metrics[class_name] = {
                "precision": float(class_metrics["precision"]),
                "recall": float(class_metrics["recall"]),
                "f1_score": float(class_metrics["f1-score"]),
                "support": int(class_metrics["support"]),
            }

        return per_class_metrics

    def _save_confusion_matrix(
        self, model_name: str, true_np: np.ndarray, pred_np: np.ndarray
    ) -> str:
        os.makedirs(self.config.confusion_matrix_dir, exist_ok=True)
        class_names = training_pipeline.DISEASE_CLASS_NAMES
        cm = confusion_matrix(true_np, pred_np, labels=list(range(len(class_names))))

        cm_path = os.path.join(
            self.config.confusion_matrix_dir, f"{model_name}_confusion_matrix.csv"
        )
        with open(cm_path, "w", encoding="utf-8") as fh:
            fh.write("actual/predicted," + ",".join(class_names) + "\n")
            for idx, row in enumerate(cm):
                fh.write(class_names[idx] + "," + ",".join(str(v) for v in row) + "\n")

        logging.info(f"Disease confusion matrix saved: {cm_path}")
        return cm_path

    def _select_model(
        self, candidate_evaluations: list[dict]
    ) -> dict:
        """
        Select the model with highest validation macro-F1 score.
        Accuracy is used as a tie-breaker.
        """
        selected = max(
            candidate_evaluations,
            key=lambda item: (
                item["val_metrics"].f1_score,
                item["val_metrics"].accuracy,
            ),
        )
        logging.info(
            f"Disease model selected: {selected['artifact'].model_name} "
            f"(val macro-F1={selected['val_metrics'].f1_score:.4f})"
        )
        return selected

    def initiate_model_evaluation(
        self,
        model_trainer_artifact: DiseaseModelTrainerArtifact,
        data_transformation_artifact: DiseaseDataTransformationArtifact,
    ) -> DiseaseModelEvaluationArtifact:
        logging.info("Starting Disease Model Evaluation.")
        try:
            val_ds = self._create_dataset(
                data_transformation_artifact.validation_data_dir, data_transformation_artifact
            )
            test_ds = self._create_dataset(
                data_transformation_artifact.test_data_dir, data_transformation_artifact
            )

            candidate_evals = []

            for trained_artifact in model_trainer_artifact.trained_model_artifacts:
                logging.info(f"Evaluating on validation: {trained_artifact.model_name}")
                model = tf.keras.models.load_model(trained_artifact.model_file_path)
                val_true, val_pred = self._get_predictions(model, val_ds)
                val_metrics = self._calculate_metrics(val_true, val_pred)

                candidate_evals.append({
                    "artifact": trained_artifact,
                    "model": model,
                    "val_metrics": val_metrics,
                })

            # Select best model on validation data only
            best_candidate = self._select_model(candidate_evals)
            selected_artifact: DiseaseTrainedModelArtifact = best_candidate["artifact"]
            selected_model: tf.keras.Model = best_candidate["model"]

            # Evaluate selected model ONCE on the untouched test set
            logging.info(f"Evaluating selected model on test set: {selected_artifact.model_name}")
            test_true, test_pred = self._get_predictions(selected_model, test_ds)
            test_metrics = self._calculate_metrics(test_true, test_pred)
            per_class_metrics = self._calculate_per_class_metrics(test_true, test_pred)

            cm_path = self._save_confusion_matrix(
                selected_artifact.model_name, test_true, test_pred
            )

            os.makedirs(self.config.evaluation_report_dir, exist_ok=True)
            report = {
                "class_names": training_pipeline.DISEASE_CLASS_NAMES,
                "selected_model_name": selected_artifact.model_name,
                "selected_model_file_path": selected_artifact.model_file_path,
                "validation_metrics": {
                    "accuracy": best_candidate["val_metrics"].accuracy,
                    "macro_precision": best_candidate["val_metrics"].precision,
                    "macro_recall": best_candidate["val_metrics"].recall,
                    "macro_f1": best_candidate["val_metrics"].f1_score,
                },
                "test_metrics": {
                    "accuracy": test_metrics.accuracy,
                    "macro_precision": test_metrics.precision,
                    "macro_recall": test_metrics.recall,
                    "macro_f1": test_metrics.f1_score,
                },
                "per_class_metrics": per_class_metrics,
                "confusion_matrix_file_path": cm_path,
            }

            with open(self.config.evaluation_report_file_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=4)

            eval_result = DiseaseModelEvaluationResult(
                model_name=selected_artifact.model_name,
                model_file_path=selected_artifact.model_file_path,
                test_metric_artifact=test_metrics,
                per_class_metrics=per_class_metrics,
                confusion_matrix_file_path=cm_path,
                is_accepted=True,
            )

            evaluation_artifact = DiseaseModelEvaluationArtifact(
                evaluated_model_artifacts=[eval_result],
                selected_model_name=selected_artifact.model_name,
                selected_model_file_path=selected_artifact.model_file_path,
                evaluation_report_file_path=self.config.evaluation_report_file_path,
            )

            logging.info(f"Disease Model Evaluation completed: {evaluation_artifact}")
            return evaluation_artifact

        except Exception as e:
            logging.error("Error in Disease Model Evaluation stage.")
            raise EthiopianCoffeeException(e, sys) from e
