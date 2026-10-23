import json
import os
import sys
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.entity.config_entity import DetectorModelEvaluationConfig
from EthiopianCoffee.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DetectorDataTransformationArtifact,
    DetectorModelEvaluationArtifact,
    DetectorModelEvaluationResult,
    DetectorModelTrainerArtifact,
    DetectorTrainedModelArtifact,
)
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging


class ModelEvaluation:
    """
    Evaluates trained binary coffee-leaf detector models.

    Evaluation flow:
      1. Create validation and test datasets.
      2. For each trained model, generate validation predictions.
      3. Tune the decision threshold on validation data ONLY (never test data).
      4. Select the best model using validation F1 score (accuracy as tie-breaker).
      5. Evaluate the selected model on the untouched test set using the tuned threshold.
      6. Save the confusion matrix and evaluation report.
    """

    def __init__(self, model_evaluation_config: DetectorModelEvaluationConfig) -> None:
        self.config = model_evaluation_config

    def _create_dataset(
        self, data_dir: str, data_transformation_artifact: DetectorDataTransformationArtifact
    ) -> tf.data.Dataset:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="int",
            class_names=training_pipeline.DETECTOR_CLASS_NAMES,
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
        probabilities: list[float] = []

        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            probabilities.extend(preds.flatten().tolist())
            true_labels.extend(labels.numpy().tolist())

        return np.array(true_labels, dtype=np.int32), np.array(probabilities, dtype=np.float32)

    def _select_threshold(self, true_np: np.ndarray, prob_np: np.ndarray) -> float:
        """
        Search over candidate thresholds on validation data.
        Selects threshold that maximizes F1 score.
        Uses ONLY validation data - never test data.
        """
        best_threshold = training_pipeline.DETECTOR_DEFAULT_DECISION_THRESHOLD
        best_f1 = 0.0

        for t in np.arange(
            training_pipeline.DETECTOR_THRESHOLD_SEARCH_MIN,
            training_pipeline.DETECTOR_THRESHOLD_SEARCH_MAX,
            training_pipeline.DETECTOR_THRESHOLD_SEARCH_STEP,
        ):
            threshold = round(float(t), 2)
            pred = (prob_np >= threshold).astype(np.int32)
            score = float(f1_score(true_np, pred, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        logging.info(
            f"Detector threshold selection (validation F1): "
            f"threshold={best_threshold:.2f}, F1={best_f1:.4f}"
        )
        return best_threshold

    def _calculate_metrics(
        self, true_np: np.ndarray, prob_np: np.ndarray, threshold: float
    ) -> ClassificationMetricArtifact:
        pred_np = (prob_np >= threshold).astype(np.int32)

        acc = float(accuracy_score(true_np, pred_np))
        prec = float(precision_score(true_np, pred_np, zero_division=0))
        rec = float(recall_score(true_np, pred_np, zero_division=0))
        f1 = float(f1_score(true_np, pred_np, zero_division=0))

        try:
            auc = float(roc_auc_score(true_np, prob_np))
        except Exception:
            auc = 0.0

        cm = confusion_matrix(true_np, pred_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return ClassificationMetricArtifact(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            auc=auc,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
        )

    def _save_confusion_matrix(
        self, model_name: str, true_np: np.ndarray, prob_np: np.ndarray, threshold: float
    ) -> str:
        os.makedirs(self.config.confusion_matrix_dir, exist_ok=True)
        pred_np = (prob_np >= threshold).astype(np.int32)
        cm = confusion_matrix(true_np, pred_np, labels=[0, 1])

        cm_path = os.path.join(
            self.config.confusion_matrix_dir, f"{model_name}_confusion_matrix.csv"
        )
        with open(cm_path, "w", encoding="utf-8") as fh:
            fh.write("actual/predicted," + ",".join(training_pipeline.DETECTOR_CLASS_NAMES) + "\n")
            for idx, row in enumerate(cm):
                fh.write(training_pipeline.DETECTOR_CLASS_NAMES[idx] + "," + ",".join(str(v) for v in row) + "\n")

        logging.info(f"Detector confusion matrix saved: {cm_path}")
        return cm_path

    def _select_model(
        self, candidate_evaluations: list[dict]
    ) -> dict:
        """
        Select model with highest validation F1 score. Accuracy as tie-breaker.
        """
        selected = max(
            candidate_evaluations,
            key=lambda item: (
                item["val_metrics"].f1_score,
                item["val_metrics"].accuracy,
            ),
        )
        logging.info(
            f"Detector model selected: {selected['artifact'].model_name} "
            f"(val F1={selected['val_metrics'].f1_score:.4f}, threshold={selected['threshold']:.2f})"
        )
        return selected

    def initiate_model_evaluation(
        self,
        model_trainer_artifact: DetectorModelTrainerArtifact,
        data_transformation_artifact: DetectorDataTransformationArtifact,
    ) -> DetectorModelEvaluationArtifact:
        logging.info("Starting Detector Model Evaluation.")
        try:
            val_ds = self._create_dataset(
                data_transformation_artifact.validation_data_dir, data_transformation_artifact
            )
            test_ds = self._create_dataset(
                data_transformation_artifact.test_data_dir, data_transformation_artifact
            )

            candidate_evals = []

            for trained_artifact in model_trainer_artifact.trained_model_artifacts:
                logging.info(f"Tuning threshold and evaluating on validation: {trained_artifact.model_name}")
                model = tf.keras.models.load_model(trained_artifact.model_file_path)
                val_true, val_prob = self._get_predictions(model, val_ds)

                tuned_threshold = self._select_threshold(val_true, val_prob)
                val_metrics = self._calculate_metrics(val_true, val_prob, tuned_threshold)

                candidate_evals.append({
                    "artifact": trained_artifact,
                    "model": model,
                    "threshold": tuned_threshold,
                    "val_metrics": val_metrics,
                })

            # Select best model on validation data
            best_candidate = self._select_model(candidate_evals)
            selected_artifact: DetectorTrainedModelArtifact = best_candidate["artifact"]
            selected_model: tf.keras.Model = best_candidate["model"]
            selected_threshold: float = best_candidate["threshold"]
            selected_val_metrics: ClassificationMetricArtifact = best_candidate["val_metrics"]

            # Evaluate selected model ONCE on test set using the tuned threshold
            logging.info(f"Evaluating selected model on test set: {selected_artifact.model_name}")
            test_true, test_prob = self._get_predictions(selected_model, test_ds)
            test_metrics = self._calculate_metrics(test_true, test_prob, selected_threshold)

            cm_path = self._save_confusion_matrix(
                selected_artifact.model_name, test_true, test_prob, selected_threshold
            )

            per_class_metrics = {
                "not_coffee_leaf": {
                    "precision": test_metrics.precision,
                    "recall": test_metrics.recall,
                    "f1_score": test_metrics.f1_score,
                },
                "coffee_leaf": {
                    "precision": test_metrics.precision,
                    "recall": test_metrics.recall,
                    "f1_score": test_metrics.f1_score,
                },
            }

            os.makedirs(self.config.evaluation_report_dir, exist_ok=True)
            report = {
                "selected_model_name": selected_artifact.model_name,
                "selected_model_file_path": selected_artifact.model_file_path,
                "selected_decision_threshold": selected_threshold,
                "validation_metrics": {
                    "accuracy": selected_val_metrics.accuracy,
                    "precision": selected_val_metrics.precision,
                    "recall": selected_val_metrics.recall,
                    "f1_score": selected_val_metrics.f1_score,
                    "auc": selected_val_metrics.auc,
                },
                "test_metrics": {
                    "accuracy": test_metrics.accuracy,
                    "precision": test_metrics.precision,
                    "recall": test_metrics.recall,
                    "f1_score": test_metrics.f1_score,
                    "auc": test_metrics.auc,
                    "true_positives": test_metrics.true_positives,
                    "true_negatives": test_metrics.true_negatives,
                    "false_positives": test_metrics.false_positives,
                    "false_negatives": test_metrics.false_negatives,
                },
                "confusion_matrix_file_path": cm_path,
            }

            with open(self.config.evaluation_report_file_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=4)

            eval_result = DetectorModelEvaluationResult(
                model_name=selected_artifact.model_name,
                model_file_path=selected_artifact.model_file_path,
                validation_metric_artifact=selected_val_metrics,
                test_metric_artifact=test_metrics,
                per_class_metrics=per_class_metrics,
                confusion_matrix_file_path=cm_path,
                decision_threshold=selected_threshold,
                is_accepted=True,
            )

            evaluation_artifact = DetectorModelEvaluationArtifact(
                evaluated_model_result=eval_result,
                selected_model_name=selected_artifact.model_name,
                selected_model_file_path=selected_artifact.model_file_path,
                selected_decision_threshold=selected_threshold,
                evaluation_report_file_path=self.config.evaluation_report_file_path,
            )

            logging.info(f"Detector Model Evaluation completed: {evaluation_artifact}")
            return evaluation_artifact

        except Exception as e:
            logging.error("Error in Detector Model Evaluation stage.")
            raise EthiopianCoffeeException(e, sys) from e
