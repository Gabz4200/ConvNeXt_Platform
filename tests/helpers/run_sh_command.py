"""Shell command runner helper for pytest suites."""

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import importlib

    sh = importlib.import_module("sh")  # type: ignore[import-not-found]


def run_sh_command(command: list[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg)


__all__ = ["run_sh_command"]
