"""Securely load the Phase 7A model artifact from Streamlit Secrets."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.inference import load_model_artifact

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
ARTIFACT_B64_SECRET = "MODEL_ARTIFACT_B64"
ARTIFACT_SHA256_SECRET = "MODEL_ARTIFACT_SHA256"


def load_artifact_from_secrets(secrets: Mapping[str, Any]) -> dict[str, Any]:
    """Verify and load a trusted joblib artifact supplied through app secrets.

    The artifact is never accepted from an app visitor or Git. SHA-256 is
    checked before joblib deserialization, and the temporary file is removed
    whether loading succeeds or fails.
    """
    try:
        encoded = secrets[ARTIFACT_B64_SECRET]
        expected_digest = secrets[ARTIFACT_SHA256_SECRET]
    except (KeyError, FileNotFoundError) as exc:
        raise ValueError(
            "Required model artifact secrets are missing or unavailable."
        ) from exc

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("The model artifact secret must be a non-empty Base64 string.")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_digest
    ):
        raise ValueError("The model artifact SHA-256 secret must contain 64 hex characters.")

    max_encoded_chars = 4 * ((MAX_ARTIFACT_BYTES + 2) // 3)
    if len(encoded) > max_encoded_chars:
        raise ValueError("The encoded model artifact exceeds the configured size limit.")

    try:
        artifact_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The model artifact secret is not valid Base64.") from exc

    if not artifact_bytes or len(artifact_bytes) > MAX_ARTIFACT_BYTES:
        raise ValueError("The decoded model artifact is empty or exceeds the size limit.")

    actual_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest.lower()):
        raise ValueError("The model artifact checksum does not match.")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="customer-churn-model-", suffix=".joblib"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(artifact_bytes)
        return load_model_artifact(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
