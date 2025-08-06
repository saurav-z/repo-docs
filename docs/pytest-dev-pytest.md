<!-- pytest Logo -->
<div align="center">
  <a href="https://docs.pytest.org/en/stable/">
    <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest" height="200">
  </a>
</div>

<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/pytest/">
    <img src="https://img.shields.io/pypi/v/pytest.svg" alt="PyPI version">
  </a>
  <a href="https://anaconda.org/conda-forge/pytest">
    <img src="https://img.shields.io/conda/vn/conda-forge/pytest.svg" alt="Conda version">
  </a>
  <a href="https://pypi.org/project/pytest/">
    <img src="https://img.shields.io/pypi/pyversions/pytest.svg" alt="Python versions">
  </a>
  <a href="https://codecov.io/gh/pytest-dev/pytest">
    <img src="https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg" alt="Code coverage">
  </a>
  <a href="https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest">
    <img src="https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg" alt="Build Status">
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

**pytest is a mature and feature-rich Python testing framework that helps you write better tests more efficiently.**

## Key Features

*   **Clear and Concise Assertions:** Provides detailed information for failing `assert statements <https://docs.pytest.org/en/stable/how_to/assert.html>`_ to pinpoint issues quickly.
*   **Automatic Test Discovery:**  Effortlessly discovers test modules and functions, saving you time and effort.
*   **Modular Fixtures:** Offers flexible fixtures for managing test resources, enabling clean and reusable test setups.
*   **Unittest Compatibility:** Seamlessly runs `unittest <https://docs.pytest.org/en/stable/how_to/unittest.html>`_ test suites.
*   **Python Compatibility:** Supports Python 3.9+ and PyPy3.
*   **Extensible with Plugins:** Boasts a rich plugin ecosystem with 1300+ `external plugins <https://docs.pytest.org/en/latest/reference/plugin_list.html>`_ to customize and extend pytest's functionality.

## Getting Started

Here's a simple example to get you started:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

Run your tests with:

```bash
pytest
```

pytest will show you the results, and, in the case of a failure, provide detailed information to help you debug.

## Documentation

Comprehensive documentation, including installation guides, tutorials, and more, can be found at:  [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

##  Bugs and Feature Requests

Please report bugs or request features via the `GitHub issue tracker <https://github.com/pytest-dev/pytest/issues>`.

## Changelog

Review the `Changelog <https://docs.pytest.org/en/stable/changelog.html>` for information on releases and enhancements.

## Support pytest

Support the continued development of pytest through the [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

pytest is available as part of the Tidelift Subscription. Get commercial support and maintenance from the maintainers of pytest and other open source dependencies. Learn more at:  <https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo>

## Security

If you find a security vulnerability, please report it via the `Tidelift security contact <https://tidelift.com/security>`.

## License

pytest is distributed under the [MIT License](https://github.com/pytest-dev/pytest/blob/main/LICENSE).

[Back to the Project Repository](https://github.com/pytest-dev/pytest)