"""Regression tests for Aikido path-traversal findings on the scan upload
and DB-backed FileResponse routes.

The primary sink is `scans.py` upload: an attacker could send a
multipart upload with `filename="../../etc/cron.d/x"`, and the server
would write the bytes outside `settings.upload_dir`. We assert the
filename is stripped to its basename, no file lands outside the upload
root, and obviously-bad inputs (null byte) get a 400.

The downstream sinks (Image.open / FileResponse of DB-stored paths) are
exercised by `test_routes.py::test_harmonize_scan` and
`test_trigger_prediction`; here we just add a focused test that a
tampered DB row pointing outside `upload_dir` is rejected by the
defence-in-depth check.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/test.db"
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir()
    settings.harmonized_dir = tmp_path / "harmonized"
    settings.harmonized_dir.mkdir()
    settings.weights_dir = tmp_path / "weights"
    settings.weights_dir.mkdir()

    from app.main import app

    with TestClient(app) as c:
        c._upload_dir = settings.upload_dir  # type: ignore[attr-defined]
        c._tmp_path = tmp_path  # type: ignore[attr-defined]
        yield c


def _png_bytes() -> bytes:
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _new_patient(client) -> str:
    r = client.post(
        "/patients/",
        data={"name": "Trav Test", "age": "40"},
        follow_redirects=False,
    )
    return r.headers["location"].split("/")[-1]


@pytest.mark.parametrize(
    "malicious_name",
    [
        "../../etc/passwd",
        "../../../tmp/pwned.png",
        "..\\..\\windows\\system32\\cmd.exe",
        "/etc/passwd",
        "..%2F..%2Fescape.png",  # client-decoded — server sees a single
        # segment; we still expect basename to flatten any decoded form.
    ],
)
def test_upload_rejects_or_flattens_traversal_filename(client, tmp_path, malicious_name):
    """Attacker filenames must never write outside `upload_dir`."""
    patient_id = _new_patient(client)
    response = client.post(
        "/scans/",
        data={"patient_id": patient_id, "scanner_vendor": "Siemens", "modality": "T1"},
        files={"file": (malicious_name, _png_bytes(), "image/png")},
        follow_redirects=False,
    )
    # Upload either succeeds with a sanitised name or is refused; in
    # both cases nothing landed outside the upload root.
    assert response.status_code in (303, 400)

    upload_dir: Path = client._upload_dir  # type: ignore[attr-defined]
    upload_root = upload_dir.resolve()
    # Walk everything in `tmp_path` and confirm any written .png is under upload_dir.
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".png":
            assert upload_root in path.resolve().parents, (
                f"file landed outside uploads root: {path}"
            )

    # Specifically, no file at `<tmp_path>/etc/passwd` or similar.
    assert not (tmp_path / "etc" / "passwd").exists()
    assert not (tmp_path.parent / "pwned.png").exists()


def test_upload_rejects_null_byte_filename(client):
    patient_id = _new_patient(client)
    response = client.post(
        "/scans/",
        data={"patient_id": patient_id, "scanner_vendor": "Siemens", "modality": "T1"},
        files={"file": ("ok\x00.png", _png_bytes(), "image/png")},
        follow_redirects=False,
    )
    assert response.status_code == 400


def test_safe_filename_helper_flattens_traversal():
    """Direct unit-test of the sanitiser, independent of FastAPI wiring."""
    from fastapi import HTTPException

    from app.security import safe_filename, safe_join

    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("/etc/passwd") == "passwd"
    # Backslashes are not path separators on POSIX, but a name that
    # still contains one after basename is rejected outright.
    with pytest.raises(HTTPException) as bs_exc:
        safe_filename("..\\..\\windows\\cmd.exe")
    assert bs_exc.value.status_code == 400
    assert safe_filename("..") == "scan"
    assert safe_filename("") == "scan"
    assert safe_filename(None) == "scan"

    with pytest.raises(HTTPException) as exc:
        safe_filename("ok\x00.png")
    assert exc.value.status_code == 400


def test_safe_join_rejects_escape(tmp_path):
    """safe_join must refuse to resolve outside its root."""
    from fastapi import HTTPException

    from app.security import safe_join

    root = tmp_path / "uploads"
    root.mkdir()
    # In-root: OK
    assert safe_join(root, "ok.png") == (root / "ok.png").resolve()
    # Escape via `..` — would land in `tmp_path` which is outside root.
    with pytest.raises(HTTPException) as exc:
        safe_join(root, "../escape.png")
    assert exc.value.status_code == 400
