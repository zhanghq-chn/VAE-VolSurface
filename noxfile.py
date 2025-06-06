from __future__ import annotations

import nox

nox.options.sessions = ["lint", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    # session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs, external=True)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    # session.install(".[test]")
    session.run("pytest", *session.posargs, external=True)


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    session.install("build")
    session.run("python", "-m", "build")
