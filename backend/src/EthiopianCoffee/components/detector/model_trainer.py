import os
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from EthiopianCoffee.constants import training_pipeline
from EthiopianCoffee.Exceptions.Exception import EthiopianCoffeeException
from EthiopianCoffee.Logging.logger import logging
from EthiopianCoffee.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DetectorDataTransformationArtifact,
    DetectorModelTrainerArtifact,
    DetectorTrainedModelArtifact,
)
from EthiopianCoffee.entity.config_entity import DetectorModelTrainerConfig
from EthiopianCoffee.models.Coffee_Detector.Coffee_Detector_Cnn_model import CNNModel
from EthiopianCoffee.models.Coffee_Detector.Coffee_Detector_transfer_learning_model import (
    TransferLearningModel,
)


class ModelTrainer:
    """
    Model Trainer component for the Coffee Leaf Detector.

    Trains both candidate models sequentially:
      1. CNN from scratch
      2. EfficientNetB0 (Phase 1: head only -> Phase 2: fine-tuning)

    Uses: train dataset + validation dataset. Test set is NEVER touched here.
    """

    def __init__(self, model_trainer_config: DetectorModelTrainerConfig) -> None:
        self.config = model_trainer_config

    def _create_datasets(
        self, data_transformation_artifact: DetectorDataTransformationArtifact
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        cfg = data_transformation_artifact
        class_names = training_pipeline.DETECTOR_CLASS_NAMES

        train_ds = tf.keras.utils.image_dataset_from_directory(
            cfg.train_data_dir,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=cfg.image_size,
            color_mode="rgb",
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            cfg.validation_data_dir,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=cfg.image_size,
            color_mode="rgb",
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        return train_ds, val_ds

    def _compute_class_weights(self, train_ds: tf.data.Dataset) -> dict[int, float]:
        labels: list[int] = []
        for _, batch_labels in train_ds:
            labels.extend(batch_labels.numpy().tolist())

        counter = Counter(labels)
        total = sum(counter.values())
        n_classes = len(counter)

        weights = {
            cls: total / (n_classes * count)
            for cls, count in counter.items()
        }
        logging.info(f"Detector class weights: {weights}")
        return weights

    def _prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def _create_callbacks(self, model_name: str) -> list[tf.keras.callbacks.Callback]:
        callback_dir = os.path.join(self.config.model_trainer_dir, "callbacks")
        os.makedirs(callback_dir, exist_ok=True)
        checkpoint_path = os.path.join(callback_dir, f"{model_name}_best.keras")

        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
            ),
        ]

    def _evaluate_on_validation(
        self, model: tf.keras.Model, val_ds: tf.data.Dataset
    ) -> ClassificationMetricArtifact:
        true_labels: list[int] = []
        prob_list: list[float] = []

        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            prob_list.extend(preds.flatten().tolist())
            true_labels.extend(labels.numpy().tolist())

        true_np = np.array(true_labels)
        prob_np = np.array(prob_list)
        pred_np = (prob_np >= training_pipeline.DETECTOR_DEFAULT_DECISION_THRESHOLD).astype(int)

        acc = float(accuracy_score(true_np, pred_np))
        prec = float(precision_score(true_np, pred_np, zero_division=0))
        rec = float(recall_score(true_np, pred_np, zero_division=0))
        f1 = float(f1_score(true_np, pred_np, zero_division=0))

        try:
            auc = float(roc_auc_score(true_np, prob_np))
        except Exception:
            auc = 0.0

        return ClassificationMetricArtifact(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            auc=auc,
        )

    def _train_cnn(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict[int, float],
    ) -> DetectorTrainedModelArtifact:
        logging.info("--- Training Detector CNN from scratch ---")
        cnn_builder = CNNModel(
            image_size=self.config.image_size,
            image_channels=self.config.image_channels,
        )
        model = cnn_builder.build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = self._create_callbacks("detector_cnn")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        os.makedirs(self.config.trained_model_dir, exist_ok=True)
        model.save(self.config.cnn_model_file_path)
        logging.info(f"Detector CNN saved to: {self.config.cnn_model_file_path}")

        val_metrics = self._evaluate_on_validation(model, val_ds)
        logging.info(f"Detector CNN validation metrics: {val_metrics}")

        return DetectorTrainedModelArtifact(
            model_name=training_pipeline.DETECTOR_CNN_MODEL_NAME,
            model_file_path=self.config.cnn_model_file_path,
            validation_metric_artifact=val_metrics,
            decision_threshold=training_pipeline.DETECTOR_DEFAULT_DECISION_THRESHOLD,
        )

    def _train_transfer_learning(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict[int, float],
    ) -> DetectorTrainedModelArtifact:
        logging.info("--- Training Detector EfficientNetB0 (Phase 1: Feature Extraction) ---")
        tl_builder = TransferLearningModel(
            image_size=self.config.image_size,
            image_channels=self.config.image_channels,
        )
        model = tl_builder.build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callbacks_p1 = self._create_callbacks("detector_efficientnetb0_p1")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks_p1,
            class_weight=class_weights,
            verbose=1,
        )

        logging.info("--- Fine-tuning Detector EfficientNetB0 (Phase 2) ---")
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is not None:
            base_model.trainable = True
            unfreeze_from = int(len(base_model.layers) * (1.0 - self.config.fine_tune_unfreeze_ratio))
            for layer in base_model.layers[:unfreeze_from]:
                layer.trainable = False
            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.fine_tune_learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            callbacks_p2 = self._create_callbacks("detector_efficientnetb0_p2")
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.fine_tune_epochs,
                callbacks=callbacks_p2,
                class_weight=class_weights,
                verbose=1,
            )

        os.makedirs(self.config.trained_model_dir, exist_ok=True)
        model.save(self.config.transfer_learning_model_file_path)
        logging.info(f"Detector EfficientNetB0 saved to: {self.config.transfer_learning_model_file_path}")

        val_metrics = self._evaluate_on_validation(model, val_ds)
        logging.info(f"Detector EfficientNetB0 validation metrics: {val_metrics}")

        return DetectorTrainedModelArtifact(
            model_name=training_pipeline.DETECTOR_TRANSFER_LEARNING_MODEL_NAME,
            model_file_path=self.config.transfer_learning_model_file_path,
            validation_metric_artifact=val_metrics,
            decision_threshold=training_pipeline.DETECTOR_DEFAULT_DECISION_THRESHOLD,
        )

    def initiate_model_training(
        self, data_transformation_artifact: DetectorDataTransformationArtifact
    ) -> DetectorModelTrainerArtifact:
        logging.info("Starting Detector Model Trainer.")
        try:
            train_ds, val_ds = self._create_datasets(data_transformation_artifact)
            class_weights = self._compute_class_weights(train_ds)

            train_ds = self._prepare_dataset(train_ds)
            val_ds = self._prepare_dataset(val_ds)

            cnn_artifact = self._train_cnn(train_ds, val_ds, class_weights)
            tl_artifact = self._train_transfer_learning(train_ds, val_ds, class_weights)

            trainer_artifact = DetectorModelTrainerArtifact(
                trained_model_artifacts=[cnn_artifact, tl_artifact]
            )
            logging.info(f"Detector Model Trainer completed: {trainer_artifact}")
            return trainer_artifact

        except Exception as e:
            logging.error("Error in Detector Model Trainer stage.")
            raise EthiopianCoffeeException(e, sys) from e
