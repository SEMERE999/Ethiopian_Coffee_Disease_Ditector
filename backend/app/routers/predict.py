from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app import config
from app.knowledge.content import DEFAULT_LANGUAGE, LANGUAGES
from app.schemas import AdvisoryReport, DetectorResult, DiseaseResult, PredictionResponse
from app.services.prediction_service import prediction_service

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="A JPEG/PNG/WEBP photo of a single coffee leaf."),
    lang: str = Form(DEFAULT_LANGUAGE, description="Report language: en, am, ti, or om."),
) -> PredictionResponse:
    if image.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. Please upload a JPEG, PNG, or WEBP image.",
        )

    if lang not in LANGUAGES:
        lang = DEFAULT_LANGUAGE

    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await image.read(1024 * 1024):
                total += len(chunk)
                if total > config.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds the {config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB upload limit.",
                    )
                tmp.write(chunk)

        try:
            result = prediction_service.predict(tmp_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"Model artifact missing: {e}") from e
        except Exception as e:  # noqa: BLE001 — surfaced to client as a clean 422
            raise HTTPException(status_code=422, detail=f"Could not process this image: {e}") from e

        report_dict = prediction_service.build_report(result.class_key, lang)

        disease_result = None
        if result.disease_predicted_class is not None:
            disease_result = DiseaseResult(
                predicted_class=result.disease_predicted_class,
                confidence=result.disease_confidence,
                probabilities=result.disease_probabilities,
            )

        return PredictionResponse(
            request_id=result.request_id,
            language=lang,
            detector=DetectorResult(
                is_coffee_leaf=result.is_coffee_leaf,
                confidence=result.detector_confidence,
                probabilities=result.detector_probabilities,
            ),
            disease=disease_result,
            class_key=result.class_key,
            report=AdvisoryReport(**report_dict),
            model_ready=result.model_ready,
            demo_mode=result.demo_mode,
            notice=result.notice,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
