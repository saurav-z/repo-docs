<p align="center">
  <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" width="200">
</p>

<!-- Badges - keep these at the top for visibility -->
[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![GitHub Actions status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![CodeTriage](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

# pytest: Powerful Python Testing Made Easy

**pytest** is a versatile and feature-rich testing framework for Python, making it simple to write effective tests that scale from small projects to complex applications. [Explore the pytest repository on GitHub](https://github.com/pytest-dev/pytest) to see its power in action.

## Key Features of pytest:

*   **Clear and Concise Assertions:**  Provides detailed information for failing assertions, eliminating the need for `self.assert*` methods.
*   **Automatic Test Discovery:**  Intelligently finds and runs your test modules and functions, saving you time and effort.
*   **Modular Fixtures:** Offers a flexible and reusable fixture system for managing test setup, teardown, and data.
*   **Compatibility:** Seamlessly integrates with `unittest` and `trial` test suites.
*   **Python Version Support:** Supports Python 3.9+ and PyPy3.
*   **Extensible Architecture:** Boasts a rich plugin ecosystem with 1300+ external plugins, fostering a vibrant and active community.

## Getting Started

Here's a quick example of a simple test:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

To run this test:

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

Learn more about getting started at the [pytest documentation](https://docs.pytest.org/en/stable/getting-started.html#our-first-test-run).

## Documentation

For comprehensive documentation, tutorials, and installation instructions, please visit the official [pytest documentation](https://docs.pytest.org/en/stable/).

## Contributing & Support

*   **Bugs/Feature Requests:**  Please use the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues) to report issues or suggest new features.
*   **Changelog:** Review the [Changelog](https://docs.pytest.org/en/stable/changelog.html) for detailed information about changes in each version.

## Support pytest

Support the development of pytest through [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

pytest is available as part of the Tidelift Subscription.

> The maintainers of pytest and thousands of other packages are working with Tidelift to deliver commercial support and maintenance for the open source dependencies you use to build your applications. Save time, reduce risk, and improve code health, while paying the maintainers of the exact dependencies you use.

[Learn more.](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## Security

While pytest has no known security vulnerabilities, please use the [Tidelift security contact](https://tidelift.com/security) to report any potential issues.  Tidelift will coordinate the fix and disclosure.

## License

pytest is licensed under the [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) license and is free and open-source software.