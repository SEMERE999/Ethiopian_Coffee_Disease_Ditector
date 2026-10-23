import os

APP_NAME = "Ethiopian Coffee Doctor API"
APP_VERSION = "1.0.0"

# Confidence below this triggers an "uncertain result" notice in the report.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))

# Upload limits
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(8 * 1024 * 1024)))  # 8 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# CORS: comma-separated list of allowed origins, "*" for any (fine for a free/demo deployment).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# Directory containing the static frontend (served directly by this API for
# single-service, zero-cost deployment). Can be overridden or disabled.
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "")
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "1") == "1"
