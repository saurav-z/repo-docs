<div align="center">
  <a href="https://docs.pytest.org/en/stable/">
    <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" width="200">
  </a>
</div>

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![Build Status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![Open Source Helpers](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera Chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

# pytest: The Powerful Python Testing Framework

**pytest is a leading Python testing framework that makes it easy to write simple tests, yet scales to support complex applications.**  Check out the original repo [here](https://github.com/pytest-dev/pytest).

## Key Features

*   **Detailed Assertion Introspection:** Get clear and informative error messages for failing assertions, making debugging a breeze.
*   **Automatic Test Discovery:**  pytest automatically finds and runs your tests, saving you time and effort.
*   **Modular Fixtures:** Easily manage test resources with a powerful fixture system, supporting both small and large-scale tests.
*   **Unittest Compatibility:** Seamlessly run your existing unittest or trial test suites.
*   **Broad Python Compatibility:** Supports Python 3.9+ and PyPy3.
*   **Extensive Plugin Ecosystem:** Extend pytest's functionality with over 1300+ external plugins, providing a rich and customizable testing experience.

## Quick Start

Here's a simple example to get you started:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

Run this test with:

```bash
pytest
```

You'll see a clear and concise output, even if the test fails, thanks to pytest's detailed assertion reports.

## Documentation

Comprehensive documentation, including installation guides, tutorials, and more, is available at [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Get Involved

*   **Report Bugs/Request Features:**  Use the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues) to submit bugs or request new features.
*   **Changelog:** Stay up-to-date on the latest changes and improvements by consulting the [Changelog](https://docs.pytest.org/en/stable/changelog.html).
*   **Support pytest:** Consider supporting pytest through [Open Collective](https://opencollective.com/pytest) to help sustain the project.

## pytest for Enterprise

pytest is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo), providing commercial support and maintenance.

## Security

Report security vulnerabilities through the [Tidelift security contact](https://tidelift.com/security).

## License

pytest is licensed under the [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) license.