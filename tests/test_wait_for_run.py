import pytest

from scripts import wait_for_run


def client_status(port: int, state: str = "completed") -> dict:
    return {
        "client_id": port,
        "state": state,
        "current_round": 1,
        "total_rounds": 1,
    }


@pytest.mark.parametrize(
    ("arguments", "expected_ports"),
    [
        ([], list(range(8000, 8008))),
        (["--client-count", "4"], list(range(8000, 8004))),
        (
            ["--client-count", "12", "--base-port", "9000"],
            list(range(9000, 9012)),
        ),
    ],
)
def test_monitors_all_configured_clients(monkeypatch, arguments, expected_ports):
    ports = []

    def status(port):
        ports.append(port)
        return client_status(port)

    monkeypatch.setattr(wait_for_run, "status", status)
    wait_for_run.main(arguments)

    assert ports == expected_ports


def test_retries_unavailable_api_and_waits_for_all_clients(monkeypatch):
    replies = iter(
        [
            OSError("connection refused"),
            client_status(8000),
            client_status(8001, "running"),
            client_status(8000),
            client_status(8001),
        ]
    )
    pauses = []

    def status(port):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(wait_for_run, "status", status)
    monkeypatch.setattr(wait_for_run.time, "sleep", pauses.append)
    wait_for_run.main(["--client-count", "2"])

    assert pauses == [5, 5]


def test_failed_client_exits_with_error(monkeypatch):
    monkeypatch.setattr(
        wait_for_run, "status", lambda port: client_status(port, "failed")
    )

    with pytest.raises(SystemExit, match="at least one client failed"):
        wait_for_run.main(["--client-count", "1"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--client-count", "0"],
        ["--client-count", "-1"],
        ["--base-port", "0"],
        ["--base-port", "65536"],
        ["--base-port", "65535", "--client-count", "2"],
    ],
)
def test_rejects_invalid_port_ranges_before_polling(monkeypatch, arguments):
    def status(port):
        pytest.fail("invalid configuration must not poll clients")

    monkeypatch.setattr(wait_for_run, "status", status)

    with pytest.raises(SystemExit) as error:
        wait_for_run.main(arguments)

    assert error.value.code == 2
