from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import config
from app.knowledge.content import CONTENT, DEFAULT_LANGUAGE, LANGUAGES
from app.schemas import HealthResponse, LanguageInfo, LanguagesResponse
from app.services.prediction_service import prediction_service

router = APIRouter(tags=["metadata"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        detector_model_loaded=prediction_service.detector_model_loaded(),
        disease_model_loaded=prediction_service.disease_model_loaded(),
        demo_mode=prediction_service.is_demo_mode(),
        version=config.APP_VERSION,
    )


@router.get("/languages", response_model=LanguagesResponse)
def languages() -> LanguagesResponse:
    return LanguagesResponse(
        default=DEFAULT_LANGUAGE,
        languages=[LanguageInfo(code=code, **info) for code, info in LANGUAGES.items()],
    )


@router.get("/diseases")
def diseases(lang: str = DEFAULT_LANGUAGE) -> dict:
    """
    Return the full advisory encyclopedia (all known classes) in one
    language — powers an in-app 'Coffee Disease Library' reference screen
    that farmers can browse without uploading a photo.
    """
    if lang not in LANGUAGES:
        lang = DEFAULT_LANGUAGE
    return {
        "language": lang,
        "classes": {
            key: block.get(lang) or block.get(DEFAULT_LANGUAGE)
            for key, block in CONTENT.items()
            if key != "not_coffee_leaf"
        },
    }


@router.get("/diseases/{class_key}")
def disease_detail(class_key: str, lang: str = DEFAULT_LANGUAGE) -> dict:
    if class_key not in CONTENT:
        raise HTTPException(status_code=404, detail=f"Unknown class '{class_key}'.")
    if lang not in LANGUAGES:
        lang = DEFAULT_LANGUAGE
    block = CONTENT[class_key]
    return {"class_key": class_key, "language": lang, "report": block.get(lang) or block.get(DEFAULT_LANGUAGE)}
