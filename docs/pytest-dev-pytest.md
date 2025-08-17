<div align="center">
  <a href="https://docs.pytest.org/en/stable/"><img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" height="200"></a>
</div>

<br>

# pytest: Powerful Testing Framework for Python

**pytest is a versatile and robust testing framework that makes it easy to write and maintain effective tests for your Python projects.** [Visit the original repository on GitHub](https://github.com/pytest-dev/pytest).

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code Coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![Test workflow status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![CodeTriage](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

## Key Features

*   **Simple Assertions:** Leverage standard Python `assert` statements for concise and readable tests.
*   **Detailed Error Reporting:** Get clear and informative reports with detailed assertion introspection, making debugging effortless.
*   **Automated Test Discovery:**  `Automatically discovers <https://docs.pytest.org/en/stable/explanation/goodpractices.html#python-test-discovery>`_ tests, reducing setup time.
*   **Modular Fixtures:** Utilize `fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ to manage test resources efficiently and create reusable test components.
*   **Unittest Compatibility:** Seamlessly run existing `unittest <https://docs.pytest.org/en/stable/how-to/unittest.html>`_ test suites.
*   **Plugin Ecosystem:** Extend pytest's functionality with a rich and growing ecosystem of over 1300+ `external plugins <https://docs.pytest.org/en/latest/reference/plugin_list.html>`_.
*   **Python Compatibility:** Supports Python 3.9+ and PyPy3.

## Getting Started

Here's a quick example to illustrate pytest's simplicity:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

Run the test with:

```bash
pytest
```

You'll get detailed output showing the test status. In this case, the test will fail, and pytest will show you exactly what went wrong:

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

Comprehensive documentation, including installation guides, tutorials, and API references, is available at: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Contributing & Support

*   **Bug Reports and Feature Requests:** Please use the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues) to report bugs or suggest new features.
*   **Changelog:**  Review the [Changelog](https://docs.pytest.org/en/stable/changelog.html) for release notes and updates.

## Support pytest

*   **Open Collective:** Donate to the project via [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

Available as part of the Tidelift Subscription:

The maintainers of pytest and thousands of other packages are working with Tidelift to deliver commercial support and
maintenance for the open source dependencies you use to build your applications.
Save time, reduce risk, and improve code health, while paying the maintainers of the exact dependencies you use.

*   [Learn more.](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## Security

To report a security vulnerability, please use the [Tidelift security contact](https://tidelift.com/security).  Tidelift will coordinate the fix and disclosure.

## License

pytest is licensed under the [MIT License](https://github.com/pytest-dev/pytest/blob/main/LICENSE).