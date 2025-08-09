<div align="center">
  <a href="https://docs.pytest.org/en/stable/">
    <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" height="200">
  </a>
</div>

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code Coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![Test workflow status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![Code Triage](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera Chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

# pytest: The Powerful Python Testing Framework

**pytest is a versatile and feature-rich Python testing framework that makes it easy to write and maintain tests for your projects.**  Discover more at the [pytest GitHub repository](https://github.com/pytest-dev/pytest).

## Key Features

*   **Detailed Assertion Introspection:** Get clear and concise information on failing assertions, making debugging a breeze.
*   **Automatic Test Discovery:** Automatically finds and runs your tests, saving you time and effort.
*   **Modular Fixtures:** Easily manage test resources and data using reusable and flexible fixtures.
*   **unittest Compatibility:** Seamlessly run existing `unittest` (or trial) test suites.
*   **Broad Python Support:** Compatible with Python 3.9+ and PyPy3.
*   **Extensive Plugin Ecosystem:** Extend pytest's functionality with over 1300+ external plugins developed by a thriving community.

## Getting Started

Here's a simple example to get you started:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

To run the test:

```bash
pytest
```

**Output:**

```
============================= test session starts =============================
collected 1 items

test_sample.py F

================================== FAILURES ===================================
_________________________________ test_answer _________________________________

    def test_answer():
>       assert inc(3) == 5
E       assert 4 == 5
E        +  where 4 = inc(3)

test_sample.py:5: AssertionError
========================== 1 failed in 0.04 seconds ===========================
```

## Documentation

Comprehensive documentation, including installation guides, tutorials, and more, is available at [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Reporting Bugs and Feature Requests

Please use the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues) to report bugs or request new features.

## Changelog

View the latest fixes and enhancements in the [Changelog](https://docs.pytest.org/en/stable/changelog.html).

## Support pytest

Support the project through the [Open Collective](https://opencollective.com/pytest) for one-time or monthly donations.

## pytest for enterprise

Get commercial support and maintenance for your open-source dependencies with a [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo).

## Security

Report security vulnerabilities through the [Tidelift security contact](https://tidelift.com/security).

## License

pytest is licensed under the [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) license.