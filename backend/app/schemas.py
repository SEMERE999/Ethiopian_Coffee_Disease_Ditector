from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LanguageInfo(BaseModel):
    code: str
    label: str
    native: str
    flag_emoji: str


class LanguagesResponse(BaseModel):
    default: str
    languages: list[LanguageInfo]


class AdvisoryReport(BaseModel):
    name: str
    local_name: str
    severity: str
    overview: str
    symptoms: list[str]
    causes: str
    favorable_conditions: list[str]
    immediate_actions: list[str]
    organic_treatment: list[str]
    chemical_treatment: list[str]
    prevention: list[str]
    economic_note: str
    when_to_seek_help: str


class DetectorResult(BaseModel):
    is_coffee_leaf: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]


class DiseaseResult(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    request_id: str
    language: str
    detector: DetectorResult
    disease: Optional[DiseaseResult] = None
    class_key: str
    report: AdvisoryReport
    model_ready: bool
    demo_mode: bool
    notice: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    detector_model_loaded: bool
    disease_model_loaded: bool
    demo_mode: bool
    version: str
