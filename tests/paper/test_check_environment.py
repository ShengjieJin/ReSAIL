from __future__ import annotations

import subprocess
import sys

import pytest

from scripts.prepare import check_environment as environment


def test_matching_tested_stack_reports_dependency_conflicts(monkeypatch):
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: environment.EXPECTED[name])
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "example-package requires missing-package, which is not installed.\n", ""
        ),
    )

    result = environment.check_environment()

    assert result["tested_stack_status"] == "verified"
    assert result["status"] == "dependency_conflicts"
    assert result["dependency_check"] == {
        "status": "conflicts",
        "issues": ["example-package requires missing-package, which is not installed."],
        "error": "",
    }


@pytest.mark.parametrize(("flags", "expected_exit"), [("--strict", 0), ("--strict-dependencies", 1)])
def test_strict_dependency_flag_controls_exit(monkeypatch, capsys, flags, expected_exit):
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: environment.EXPECTED[name])
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "conflict\n", ""),
    )
    monkeypatch.setattr(sys, "argv", ["check_environment.py", flags])

    assert environment.main() == expected_exit
    assert '"status": "conflicts"' in capsys.readouterr().out


def test_dependency_check_failure_is_reported(monkeypatch):
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: environment.EXPECTED[name])
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.12.3")

    def unavailable(*_args, **_kwargs):
        raise OSError("python executable unavailable")

    monkeypatch.setattr(environment.subprocess, "run", unavailable)
    result = environment.check_environment()
    assert result["tested_stack_status"] == "verified"
    assert result["status"] == "dependency_check_error"
    assert "python executable unavailable" in result["dependency_check"]["error"]


@pytest.mark.parametrize(
    ("flags", "expected_exit"),
    [([], 0), (["--strict"], 1), (["--strict-dependencies"], 0)],
)
def test_version_differences_are_advisory_unless_strict(monkeypatch, capsys, flags, expected_exit):
    monkeypatch.setattr(environment.importlib.metadata, "version", lambda name: "0.0.0")
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", ""),
    )
    monkeypatch.setattr(sys, "argv", ["check_environment.py", *flags])

    assert environment.main() == expected_exit
    output = capsys.readouterr().out
    assert '"tested_stack_status": "different_environment"' in output
    assert "python: tested 3.12.3, found 3.12.4" in output
