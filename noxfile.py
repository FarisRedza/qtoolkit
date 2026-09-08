import nox # type: ignore


PYTHON_VERSIONS = [
    '3.9',
    '3.10',
    '3.11',
    '3.12',
    '3.13',
    '3.14'
]


@nox.session(
    python=PYTHON_VERSIONS,
    venv_backend='uv',
)
def tests(session: nox.Session) -> None:
    session.install('.[dev]')
    session.run('pytest')

@nox.session(python='3.14', venv_backend='uv')
def docs(session):
    session.install('.[doc]')
    session.run(
        'sphinx-build',
        '-W',
        '-b',
        'html',
        'docs',
        'docs/_build/html',
    )