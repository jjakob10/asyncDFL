import http.client

import scripts.wait_for_run as wait_for_run


def test_monitor_retries_transient_startup_error(monkeypatch) -> None:
    calls = 0

    def fake_status(port: int) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http.client.RemoteDisconnected("starting")
        return {
            "client_id": port - 8000,
            "state": "completed",
            "current_round": 1,
            "total_rounds": 1,
        }

    monkeypatch.setattr(wait_for_run, "status", fake_status)
    monkeypatch.setattr(wait_for_run.time, "sleep", lambda seconds: None)

    wait_for_run.main()

    assert calls == 9
