<div align="center">
  <a href="https://docs.pytest.org/en/stable/">
    <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest Logo" width="200" height="200">
  </a>
</div>

[![PyPI Version](https://img.shields.io/pypi/v/pytest.svg)](https://pypi.org/project/pytest/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pytest.svg)](https://anaconda.org/conda-forge/pytest)
[![Python Versions](https://img.shields.io/pypi/pyversions/pytest.svg)](https://pypi.org/project/pytest/)
[![Code Coverage](https://codecov.io/gh/pytest-dev/pytest/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest)
[![GitHub Workflow Status](https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg)](https://github.com/pytest-dev/pytest/actions?query=workflow%3Atest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pytest-dev/pytest/main.svg)](https://results.pre-commit.ci/latest/github/pytest-dev/pytest/main)
[![CodeTriage](https://www.codetriage.com/pytest-dev/pytest/badges/users.svg)](https://www.codetriage.com/pytest-dev/pytest)
[![Documentation Status](https://readthedocs.org/projects/pytest/badge/?version=latest)](https://pytest.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-pytest--dev-blue)](https://discord.com/invite/pytest-dev)
[![Libera Chat](https://img.shields.io/badge/Libera%20chat-%23pytest-orange)](https://web.libera.chat/#pytest)

# pytest: Powerful Python Testing Framework

**pytest** is a leading Python testing framework that makes it easy to write simple tests and scale to complex functional testing for libraries and applications. [Visit the original repository on GitHub](https://github.com/pytest-dev/pytest).

## Key Features

*   **Clear and Concise Assertions:** Get detailed information on failing `assert statements <https://docs.pytest.org/en/stable/how_to/assert.html>`_, eliminating the need to remember `self.assert*` methods.
*   **Automated Test Discovery:**  `Auto-discovery <https://docs.pytest.org/en/stable/explanation/goodpractices.html#python-test-discovery>`_ of test modules and functions streamlines test execution.
*   **Flexible Fixtures:** Leverage `Modular fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ for managing test resources, whether small or parametrized, and long-lived.
*   **Unittest Compatibility:**  Seamlessly run `unittest <https://docs.pytest.org/en/stable/how_to/unittest.html>`_ test suites without modification.
*   **Python Compatibility:** Supports Python 3.9+ and PyPy3.
*   **Extensible with Plugins:** Enjoy a rich plugin ecosystem, with 1300+ `external plugins <https://docs.pytest.org/en/latest/reference/plugin_list.html>`_ that enhance and extend functionality.

## Getting Started

Here's a simple example of a pytest test:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

Run the test:

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

## Documentation

Find complete documentation, including installation guides, tutorials, and PDF documents, at https://docs.pytest.org/en/stable/.

## Reporting Issues and Feature Requests

Please submit bugs and suggest new features through the `GitHub issue tracker <https://github.com/pytest-dev/pytest/issues>`.

## Changelog

Review the `Changelog <https://docs.pytest.org/en/stable/changelog.html>`__ page for details on fixes and enhancements for each pytest release.

## Support and Contribution

*   **Support pytest:** Consider supporting the project through the `Open Collective <https://opencollective.com/pytest>`_.
*   **pytest for Enterprise:** Available as part of the Tidelift Subscription for commercial support and maintenance.  `Learn more. <https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo>`_

## Security

While pytest has no known security vulnerabilities, report any concerns via the `Tidelift security contact <https://tidelift.com/security>`. Tidelift will coordinate the fix and disclosure.

## License

pytest is distributed under the `MIT`_ license and is free and open source software. Copyright Holger Krekel and others, 2004.