<div align="center">
  <a href="https://docs.pytest.org/en/stable/"><img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest" width="200"></a>
</div>

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code Coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![Test Status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![Open Collective backers and sponsors](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

# pytest: Powerful Testing with Python

**pytest** is a leading framework that makes writing tests easy and efficient, empowering you to build robust and reliable software.

## Key Features

*   **Simple Assertions:** Use standard `assert` statements for readable and expressive tests with detailed introspection for failures.
*   **Automatic Test Discovery:** Automatically finds and runs tests, simplifying test execution.
*   **Modular Fixtures:** Create reusable fixtures to set up test environments, promoting code reuse and reducing duplication.
*   **Extensive Plugin Ecosystem:** Benefit from a rich plugin architecture with thousands of external plugins to extend pytest's functionality.
*   **unittest Compatibility:** Seamlessly run existing `unittest` test suites.
*   **Python Compatibility:** Supports Python 3.9+ and PyPy3.

## Getting Started

Here's a quick example of a simple test:

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

pytest will execute your tests and provide clear and informative results, including detailed error messages for failing assertions. For more details, see the [Getting Started Guide](https://docs.pytest.org/en/stable/getting-started.html).

## Documentation

*   **Comprehensive Documentation:**  Explore the full documentation, including installation guides, tutorials, and detailed API references, at [pytest Documentation](https://docs.pytest.org/en/stable/).

## Contributing & Support

*   **Bug Reports and Feature Requests:** Submit any bugs or request new features on the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues).
*   **Changelog:** Review the [Changelog](https://docs.pytest.org/en/stable/changelog.html) for details on each release's fixes and enhancements.
*   **Support pytest:**  Become a supporter through [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

pytest is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo). Tidelift provides commercial support and maintenance for the open source dependencies you use.

## Security

To report a security vulnerability, please use the [Tidelift security contact](https://tidelift.com/security).

## License

pytest is distributed under the [MIT License](https://github.com/pytest-dev/pytest/blob/main/LICENSE).

## Learn More

For more information, visit the [pytest GitHub repository](https://github.com/pytest-dev/pytest).