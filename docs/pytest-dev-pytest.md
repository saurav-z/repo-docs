<p align="center">
  <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" width="200">
</p>

# pytest: Powerful Testing for Python

**pytest is a leading Python testing framework that simplifies test writing and scales easily for complex applications.**  [Visit the original repository on GitHub](https://github.com/pytest-dev/pytest).

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code coverage status](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![GitHub Actions status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![CodeTriage users](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera Chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

## Key Features

*   **Clear and Concise Assertions:** Get detailed information on failing `assert statements <https://docs.pytest.org/en/stable/how_to/assert.html>`_ without the need for complex `self.assert*` methods.
*   **Automatic Test Discovery:**  `pytest <https://docs.pytest.org/en/stable/explanation/goodpractices.html#python-test-discovery>`_ automatically finds and runs your test modules and functions, saving you time and effort.
*   **Modular Fixtures:**  Utilize `modular fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ to manage test resources efficiently, supporting both small and parameterized, long-lived test environments.
*   **Unittest Compatibility:** Seamlessly run your existing `unittest <https://docs.pytest.org/en/stable/how_to/unittest.html>`_ or trial test suites.
*   **Python Version Support:** Compatible with Python 3.9+ and PyPy3.
*   **Extensive Plugin Ecosystem:** Leverage a rich plugin architecture with over 1300+ `external plugins <https://docs.pytest.org/en/latest/reference/plugin_list.html>`_ and a thriving community.

## Getting Started

Here's a simple example of how to get started with pytest:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

To run the test, simply execute:

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

pytest provides detailed assertion introspection, making it easy to understand test failures.  See the `getting-started guide <https://docs.pytest.org/en/stable/getting-started.html#our-first-test-run>`_ for more examples.

## Documentation

For comprehensive documentation, including installation instructions, tutorials, and other resources, please visit the official documentation at [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Contributing & Support

*   **Bugs and Feature Requests:**  Submit bugs or request new features using the `GitHub issue tracker <https://github.com/pytest-dev/pytest/issues>`_.
*   **Changelog:** Consult the `Changelog <https://docs.pytest.org/en/stable/changelog.html>`_ for a detailed history of changes and enhancements in each version.
*   **Support pytest:**
    *   Contribute to the project through [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

pytest is available as part of the Tidelift Subscription.  Tidelift provides commercial support and maintenance for the open-source dependencies you use.  Learn more at:  `Learn more. <https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo>`_

## Security

If you find a security vulnerability, please report it through the `Tidelift security contact <https://tidelift.com/security>`_.

## License

pytest is licensed under the terms of the `MIT`_ license and is free and open-source software.

.. _`MIT`: https://github.com/pytest-dev/pytest/blob/main/LICENSE