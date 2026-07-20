from __future__ import annotations

import pytest

from app.config import Settings


def test_settings_parse_plural_heads_and_shared_run_id(monkeypatch) -> None:
    monkeypatch.setenv("CLIENT_ID", "client-8")
    monkeypatch.setenv("CLIENT_COUNT", "10")
    monkeypatch.setenv("HEAD_CLIENT_IDS", "7,8,9")
    monkeypatch.setenv("EMPTY_CLIENT_IDS", "0,4,9")
    monkeypatch.setenv("RUN_ID", "mnist-8p2-top4")

    settings = Settings.from_env()

    assert settings.head_client_ids == (7, 8, 9)
    assert settings.is_head
    assert settings.run_id == "mnist-8p2-top4"
    assert settings.empty_client_ids == (0, 4, 9)
    assert settings.has_local_data


def test_explicit_empty_neighbors_does_not_derive_legacy_ring(monkeypatch) -> None:
    monkeypatch.setenv("CLIENT_ID", "client-8")
    monkeypatch.setenv("CLIENT_COUNT", "10")
    monkeypatch.setenv("HEAD_CLIENT_IDS", "7,8,9")
    monkeypatch.setenv("NEIGHBORS", "")

    assert Settings.from_env().neighbors == ()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("HEAD_CLIENT_IDS", "", "at least one"),
        ("HEAD_CLIENT_IDS", "7,7", "duplicate"),
        ("HEAD_CLIENT_IDS", "10", "between"),
        ("EMPTY_CLIENT_IDS", "11", "between"),
        ("RUN_ID", "", "RUN_ID"),
    ],
)
def test_settings_reject_invalid_paper_identity(
    monkeypatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv("CLIENT_ID", "client-0")
    monkeypatch.setenv("CLIENT_COUNT", "10")
    monkeypatch.setenv("HEAD_CLIENT_IDS", "7")
    monkeypatch.setenv("RUN_ID", "valid")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()
