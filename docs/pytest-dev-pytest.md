<p align="center">
  <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest Logo" width="200">
</p>

[![PyPI version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![Build Status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![CodeTriage](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera Chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

# pytest: Powerful Testing with Python

**pytest is a robust and versatile testing framework for Python that makes it easy to write simple tests while also supporting complex functional testing.**

## Key Features

*   **Clear and Concise Assertions:** Get detailed information on failing `assert statements <https://docs.pytest.org/en/stable/how_to/assert.html>`, eliminating the need to remember `self.assert*` methods.
*   **Automatic Test Discovery:**  Effortlessly find and run tests with `auto-discovery <https://docs.pytest.org/en/stable/explanation/goodpractices.html#python-test-discovery>`_ of test modules and functions.
*   **Modular Fixtures:**  Utilize `modular fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ for managing small or parametrized long-lived test resources.
*   **Unittest Compatibility:**  Seamlessly run `unittest <https://docs.pytest.org/en/stable/how_to/unittest.html>`_ (or trial) test suites out of the box.
*   **Python Version Support:** Compatible with Python 3.9+ and PyPy3.
*   **Extensible Architecture:** Benefit from a rich plugin ecosystem with over 1300+ `external plugins <https://docs.pytest.org/en/latest/reference/plugin_list.html>`_ and a vibrant community.

## Getting Started

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

To run the test, execute:

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
See `getting-started <https://docs.pytest.org/en/stable/getting-started.html#our-first-test-run>`_ for more examples.

## Documentation

Comprehensive documentation, including installation guides, tutorials, and detailed explanations, is available at:  [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)

## Reporting Bugs and Requesting Features

Please use the  `GitHub issue tracker <https://github.com/pytest-dev/pytest/issues>`_ to report bugs or suggest new features.

## Changelog

Review the `Changelog <https://docs.pytest.org/en/stable/changelog.html>`__ page to see the latest fixes and enhancements.

## Support pytest

Support the development of pytest through [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

pytest is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo).

## Security

While pytest has no known security vulnerabilities, report any concerns via the `Tidelift security contact <https://tidelift.com/security>`.

## License

pytest is licensed under the  `MIT`_ license.  It is free and open-source software.

## Contributing

We welcome contributions!  See the [pytest repository on GitHub](https://github.com/pytest-dev/pytest) for details on how to contribute.