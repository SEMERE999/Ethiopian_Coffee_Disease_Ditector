from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionInput:
    image_file_path: str


@dataclass
class PredictionOutput:
    is_coffee_leaf: bool
    detector_confidence: float
    detector_probabilities: dict[str, float]
    predicted_class: Optional[str]
    confidence: Optional[float]
    class_probabilities: Optional[dict[str, float]]
