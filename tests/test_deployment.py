import base64
import hashlib

import pytest

import src.deployment as deployment


def _secrets(payload: bytes) -> dict[str, str]:
    return {
        deployment.ARTIFACT_B64_SECRET: base64.b64encode(payload).decode("ascii"),
        deployment.ARTIFACT_SHA256_SECRET: hashlib.sha256(payload).hexdigest(),
    }


def test_load_artifact_from_secrets_verifies_and_loads_trusted_bytes(monkeypatch):
    payload = b"trusted-model-artifact"
    expected_bundle = {"model": "loaded"}
    observed = {}

    def fake_loader(path):
        observed["path"] = path
        observed["bytes"] = path.read_bytes()
        return expected_bundle

    monkeypatch.setattr(deployment, "load_model_artifact", fake_loader)
    result = deployment.load_artifact_from_secrets(_secrets(payload))

    assert result is expected_bundle
    assert observed["bytes"] == payload
    assert not observed["path"].exists()


@pytest.mark.parametrize(
    "secrets",
    [
        {},
        {deployment.ARTIFACT_B64_SECRET: "YWJj"},
        {deployment.ARTIFACT_SHA256_SECRET: "0" * 64},
    ],
)
def test_load_artifact_from_secrets_rejects_missing_secrets(secrets):
    with pytest.raises(ValueError, match="secret"):
        deployment.load_artifact_from_secrets(secrets)


@pytest.mark.parametrize("encoded", ["not-base64!", "YWJj="])
def test_load_artifact_from_secrets_rejects_malformed_base64(encoded):
    secrets = {
        deployment.ARTIFACT_B64_SECRET: encoded,
        deployment.ARTIFACT_SHA256_SECRET: "0" * 64,
    }
    with pytest.raises(ValueError, match="Base64"):
        deployment.load_artifact_from_secrets(secrets)


def test_load_artifact_from_secrets_rejects_malformed_digest():
    secrets = {
        deployment.ARTIFACT_B64_SECRET: base64.b64encode(b"artifact").decode("ascii"),
        deployment.ARTIFACT_SHA256_SECRET: "not-a-checksum",
    }
    with pytest.raises(ValueError, match="64 hex"):
        deployment.load_artifact_from_secrets(secrets)


def test_load_artifact_from_secrets_rejects_checksum_mismatch(monkeypatch):
    secrets = _secrets(b"artifact")
    secrets[deployment.ARTIFACT_SHA256_SECRET] = "0" * 64
    monkeypatch.setattr(
        deployment,
        "load_model_artifact",
        lambda path: pytest.fail("Unverified bytes must never reach joblib"),
    )
    with pytest.raises(ValueError, match="checksum"):
        deployment.load_artifact_from_secrets(secrets)


def test_load_artifact_from_secrets_rejects_empty_payload():
    secrets = _secrets(b"")
    with pytest.raises(ValueError, match="empty"):
        deployment.load_artifact_from_secrets(secrets)


def test_load_artifact_from_secrets_removes_temp_file_when_loader_fails(monkeypatch):
    payload = b"trusted-model-artifact"
    observed = {}

    def failing_loader(path):
        observed["path"] = path
        raise RuntimeError("load failed")

    monkeypatch.setattr(deployment, "load_model_artifact", failing_loader)
    with pytest.raises(RuntimeError, match="load failed"):
        deployment.load_artifact_from_secrets(_secrets(payload))
    assert not observed["path"].exists()
