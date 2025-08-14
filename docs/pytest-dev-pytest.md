<!-- pytest Logo -->
<div align="center">
  <a href="https://docs.pytest.org/en/stable/">
    <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest logo" height="200">
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
    <img src="https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg" alt="Test status">
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
    <img src="https://img.shields.io/badge/Libera%20chat-%23pytest-orange" alt="Libera chat">
  </a>
</p>

# pytest: Powerful Testing with Python

**pytest is a robust, open-source testing framework for Python, making it easy to write simple tests that scale to complex applications.** You can find the original repository [here](https://github.com/pytest-dev/pytest).

## Key Features of pytest:

*   **Clear Assertion Introspection:** Provides detailed information on failing `assert` statements, simplifying debugging and reducing the need for complex `self.assert*` methods.
*   **Automated Test Discovery:** Automatically finds and runs test modules and functions, saving you time and effort.
*   **Modular Fixtures:** Offers a powerful fixture system for managing test setup, teardown, and resource sharing, enabling efficient and reusable tests.
*   **Unittest Compatibility:** Seamlessly runs `unittest` (and trial) test suites, integrating with existing test code.
*   **Python 3.9+ & PyPy3 Support:** Compatible with modern Python versions, ensuring broad applicability.
*   **Extensible Plugin Architecture:** Features a rich ecosystem of over 1300+ external plugins, allowing for customization and integration with various tools and workflows.
*   **Comprehensive Documentation:** Extensive documentation is available for installation, tutorials, and detailed usage [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Getting Started with pytest:

A simple example of a test:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

To execute the test, simply run:

```bash
pytest
```

pytest will then run the test and provide detailed output, including any failures:

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

## Contributing and Support

*   **Bugs/Requests:** Please use the [GitHub issue tracker](https://github.com/pytest-dev/pytest/issues) to submit bugs or request features.
*   **Changelog:** Consult the [Changelog](https://docs.pytest.org/en/stable/changelog.html) page for fixes and enhancements of each version.
*   **Support:** Consider supporting pytest via [Open Collective](https://opencollective.com/pytest) to help ensure the project's continued development.

## pytest for Enterprise

pytest is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo), providing commercial support and maintenance for open-source dependencies.

## Security

If you discover a security vulnerability, please report it through the [Tidelift security contact](https://tidelift.com/security).

## License

pytest is licensed under the [MIT License](https://github.com/pytest-dev/pytest/blob/main/LICENSE).