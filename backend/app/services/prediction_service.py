"""
Service layer between the FastAPI routers and the existing
EthiopianCoffee.pipeline.PredictionPipeline (cascaded Detector -> Disease
classifier).

Design goals:
  1. Zero hard crash if the trained .keras models are not present yet
     (e.g. right after cloning the repo, before `python main.py train-all`
     has been run). Instead the API answers in a clearly-labeled
     "demo_mode", using a lightweight deterministic image heuristic, so the
     frontend and whole request/response contract can be developed, demoed
     and deployed for free before training finishes.
  2. Once real models exist under backend/final_model/..., the service
     automatically switches to genuine inference with no code change
     required.
  3. All disease-name <-> language enrichment happens here, keeping the
     ML package (src/EthiopianCoffee) untouched and reusable outside the
     web API (CLI, notebooks, batch jobs, etc.).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from app import config
from app.knowledge.content import DEFAULT_LANGUAGE, LANGUAGES, get_report

# The ML package lives at backend/src/EthiopianCoffee and is installed via
# `pip install -e backend` (see backend/setup.py), so it imports as a normal
# top-level package.
from EthiopianCoffee.constants import training_pipeline as tp


@dataclass
class PredictionResult:
    request_id: str
    class_key: str
    is_coffee_leaf: bool
    detector_confidence: float
    detector_probabilities: dict
    disease_predicted_class: Optional[str]
    disease_confidence: Optional[float]
    disease_probabilities: Optional[dict]
    model_ready: bool
    demo_mode: bool
    notice: Optional[str]


class PredictionService:
    def __init__(self) -> None:
        self._pipeline = None
        self._pipeline_import_error: Optional[str] = None
        self._detector_ready: Optional[bool] = None
        self._disease_ready: Optional[bool] = None

    # ------------------------------------------------------------
    # Model readiness
    # ------------------------------------------------------------
    def _detector_model_path(self) -> str:
        return os.path.join(tp.FINAL_MODEL_DIR, tp.DETECTOR_FINAL_MODEL_SUBDIR, tp.MODEL_FILE_NAME)

    def _disease_model_path(self) -> str:
        return os.path.join(tp.FINAL_MODEL_DIR, tp.DISEASE_FINAL_MODEL_SUBDIR, tp.MODEL_FILE_NAME)

    def detector_model_loaded(self) -> bool:
        return os.path.isfile(self._detector_model_path())

    def disease_model_loaded(self) -> bool:
        return os.path.isfile(self._disease_model_path())

    def is_demo_mode(self) -> bool:
        return not (self.detector_model_loaded() and self.disease_model_loaded())

    def _get_pipeline(self):
        if self._pipeline is None:
            from EthiopianCoffee.pipeline import PredictionPipeline  # local import: heavy (TF)

            self._pipeline = PredictionPipeline()
        return self._pipeline

    # ------------------------------------------------------------
    # Demo-mode heuristic (used only when real models are absent)
    # ------------------------------------------------------------
    @staticmethod
    def _demo_heuristic(image_path: str) -> tuple[str, dict]:
        """
        A transparent, deterministic placeholder so the full product can be
        demoed end-to-end before training finishes. NOT a real diagnosis.
        Chooses a class based on a hash of the image bytes so the same photo
        always yields the same demo result (useful for QA), spread evenly
        across the four disease classes.
        """
        with open(image_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        bucket = int(digest[:8], 16) % len(tp.DISEASE_CLASS_NAMES)
        chosen = tp.DISEASE_CLASS_NAMES[bucket]

        base = 0.55 + (int(digest[8:10], 16) / 255.0) * 0.35  # 0.55–0.90
        remainder = 1.0 - base
        others = [c for c in tp.DISEASE_CLASS_NAMES if c != chosen]
        share = remainder / len(others) if others else 0.0
        probs = {c: (base if c == chosen else share) for c in tp.DISEASE_CLASS_NAMES}
        return chosen, probs

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def predict(self, image_path: str) -> PredictionResult:
        request_id = uuid.uuid4().hex[:12]

        # Always validate the file is a decodable image before doing anything else.
        with Image.open(image_path) as img:
            img.verify()

        if self.is_demo_mode():
            chosen, probs = self._demo_heuristic(image_path)
            detector_probs = {"not_coffee_leaf": 0.08, "coffee_leaf": 0.92}
            return PredictionResult(
                request_id=request_id,
                class_key=chosen,
                is_coffee_leaf=True,
                detector_confidence=0.92,
                detector_probabilities=detector_probs,
                disease_predicted_class=chosen,
                disease_confidence=probs[chosen],
                disease_probabilities=probs,
                model_ready=False,
                demo_mode=True,
                notice=(
                    "DEMO MODE: the trained detection/diagnosis models are not deployed yet, "
                    "so this result is a placeholder for interface testing, not a real diagnosis. "
                    "Run `python main.py train-all` and redeploy to enable real predictions."
                ),
            )

        # ---- Real inference path ----
        from EthiopianCoffee.entity.prediction_entity import PredictionInput

        pipeline = self._get_pipeline()
        output = pipeline.predict(PredictionInput(image_file_path=image_path))

        if not output.is_coffee_leaf:
            return PredictionResult(
                request_id=request_id,
                class_key="not_coffee_leaf",
                is_coffee_leaf=False,
                detector_confidence=output.detector_confidence,
                detector_probabilities=output.detector_probabilities,
                disease_predicted_class=None,
                disease_confidence=None,
                disease_probabilities=None,
                model_ready=True,
                demo_mode=False,
                notice=None,
            )

        notice = None
        if output.confidence is not None and output.confidence < config.LOW_CONFIDENCE_THRESHOLD:
            notice = (
                "The model's confidence in this diagnosis is relatively low. Treat this as a "
                "preliminary indication: retake the photo in better light/focus, check a few more "
                "leaves on the same tree, and consult a local agronomist before taking major action."
            )

        return PredictionResult(
            request_id=request_id,
            class_key=output.predicted_class,
            is_coffee_leaf=True,
            detector_confidence=output.detector_confidence,
            detector_probabilities=output.detector_probabilities,
            disease_predicted_class=output.predicted_class,
            disease_confidence=output.confidence,
            disease_probabilities=output.class_probabilities,
            model_ready=True,
            demo_mode=False,
            notice=notice,
        )

    @staticmethod
    def build_report(class_key: str, lang: str) -> dict:
        if lang not in LANGUAGES:
            lang = DEFAULT_LANGUAGE
        return get_report(class_key, lang)


# Singleton reused across requests so the TF models are loaded once,
# lazily, on first real (non-demo) prediction.
prediction_service = PredictionService()
