<!-- pytest Logo -->
<p align="center">
  <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" width="200">
</p>

<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/pytest/">
    <img src="https://img.shields.io/pypi/v/pytest.svg" alt="PyPI Version">
  </a>
  <a href="https://anaconda.org/conda-forge/pytest">
    <img src="https://img.shields.io/conda/vn/conda-forge/pytest.svg" alt="Conda Version">
  </a>
  <a href="https://pypi.org/project/pytest/">
    <img src="https://img.shields.io/pypi/pyversions/pytest.svg" alt="Python Versions">
  </a>
  <a href="https://codecov.io/gh/pytest-dev/pytest">
    <img src="https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest">
    <img src="https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main">
    <img src="https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg" alt="pre-commit.ci status">
  </a>
  <a href="https://www.codetriage.com/pytest-dev/pytest">
    <img src="https://www.codetriage.com/pytest-dev/pytest/badges/users.svg" alt="Open Source Helpers">
  </a>
  <a href="https://pytest.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/pytest/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://discord.com/invite/pytest-dev">
    <img src="https://img.shields.io/badge/Discord-pytest--dev-blue" alt="Discord">
  </a>
  <a href="https://web.libera.chat/#pytest">
    <img src="https://img.shields.io/badge/Libera%20chat-%23pytest-orange" alt="Libera Chat">
  </a>
</p>

# pytest: Powerful Testing with Python

**pytest is a leading Python testing framework that makes it easy to write simple tests while supporting complex functional testing needs.**  Get started and revolutionize your testing workflow with pytest!

## Key Features

*   **Clear and Concise Assertions:** Provides detailed information on failing assertions, simplifying debugging.
*   **Automatic Test Discovery:**  Finds and runs tests automatically, saving time and effort.
*   **Modular Fixtures:** Offers flexible fixtures for managing test resources and dependencies.
*   **Unittest Compatibility:** Runs existing unittest or trial test suites seamlessly.
*   **Broad Compatibility:**  Supports Python 3.9+ and PyPy3.
*   **Extensible Architecture:** A rich plugin ecosystem with 1300+ external plugins to extend functionality.

## Getting Started

Install pytest:

```bash
pip install pytest
```

Write your first test (e.g., `test_sample.py`):

```python
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

Run your tests:

```bash
pytest
```

pytest will automatically discover and run your tests, providing clear output and detailed information about any failures.

## Documentation

Explore comprehensive documentation, including installation guides, tutorials, and more, at [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Contributing and Support

*   **Bug Reports and Feature Requests:**  Please use the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues) to report bugs or request new features.
*   **Changelog:** Review the [Changelog](https://docs.pytest.org/en/stable/changelog.html) for details on recent updates.
*   **Support pytest:** Consider supporting the project through [Open Collective](https://opencollective.com/pytest) to help sustain its development.

## pytest for Enterprise

pytest is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo). Tidelift provides commercial support and maintenance for pytest and other open-source dependencies.

## Security

Report security vulnerabilities through the [Tidelift security contact](https://tidelift.com/security).

## License

pytest is licensed under the [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) license and is free and open-source software.

**[Back to the pytest GitHub Repository](https://github.com/pytest-dev/pytest)**