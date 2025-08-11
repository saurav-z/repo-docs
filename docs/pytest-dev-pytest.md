<p align="center">
  <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" width="200">
</p>

# pytest: The Powerful Testing Framework for Python

**pytest is a leading open-source testing framework for Python, designed to make writing tests simple, efficient, and scalable.** Enhance your software development workflow with pytest's intuitive features and extensive plugin ecosystem.  [Explore the pytest GitHub repository](https://github.com/pytest-dev/pytest) to learn more.

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![CI status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![CodeTriage](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

## Key Features

*   **Clear and Detailed Assertions:**  Get easy-to-understand error messages with detailed information about failing assertions, eliminating the need for `self.assert*` methods.
*   **Automated Test Discovery:** Automatically finds and runs your test modules and functions, streamlining your testing process.
*   **Flexible Fixtures:** Utilize modular fixtures to manage test resources, enabling reuse and simplifying complex test setups.
*   **Unittest Compatibility:** Seamlessly run existing `unittest` (or trial) test suites directly with pytest.
*   **Python Version Support:** Compatible with Python 3.9+ and PyPy3.
*   **Extensive Plugin Ecosystem:** Access a rich ecosystem of over 1300+ external plugins, extending pytest's functionality to meet your specific testing needs.

## Getting Started

Here's a simple example to illustrate how easy it is to write and run tests with pytest:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

Run the test from your terminal:

```bash
$ pytest
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

pytest provides detailed assertion introspection, so you can use plain `assert` statements. For more examples, see the [Getting Started](https://docs.pytest.org/en/stable/getting-started.html#our-first-test-run) guide.

## Documentation

Comprehensive documentation, including installation guides, tutorials, and PDF documents, is available at: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)

## Reporting Issues and Feature Requests

Please submit bug reports and feature requests via the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues).

## Changelog

Review the latest fixes and enhancements in the [Changelog](https://docs.pytest.org/en/stable/changelog.html) page.

## Support pytest

Consider supporting the project through the [Open Collective](https://opencollective.com/pytest) platform with a one-time or monthly donation.

## pytest for Enterprise

pytest is available as part of the Tidelift Subscription, which provides commercial support and maintenance. Learn more about this service at:  [https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## Security

To report a security vulnerability, please contact the Tidelift security team: [https://tidelift.com/security](https://tidelift.com/security).

## License

pytest is licensed under the [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) license and is free and open-source software.