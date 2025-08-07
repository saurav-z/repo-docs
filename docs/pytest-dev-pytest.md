<!-- pytest Logo -->
<p align="center">
  <img src="https://github.com/pytest-dev/pytest/raw/main/doc/en/img/pytest_logo_curves.svg" alt="pytest Logo" height="200">
</p>

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
    <img src="https://github.com/pytest-dev/pytest/actions/workflows/test.yml/badge.svg" alt="Build Status">
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

# pytest: Powerful Python Testing Made Easy

**pytest** is a leading Python testing framework, making it simple to write effective tests and scale them for complex applications.  [Visit the original repository on GitHub](https://github.com/pytest-dev/pytest).

## Key Features

*   **Clear and Concise Assertions:** Get detailed information on failing `assert statements <https://docs.pytest.org/en/stable/how_to/assert.html>`_, eliminating the need for `self.assert*` methods.
*   **Automatic Test Discovery:** pytest automatically finds your test modules and functions, streamlining your workflow with `Auto-discovery <https://docs.pytest.org/en/stable/explanation/goodpractices.html#python-test-discovery>`_.
*   **Modular Fixtures:** Use `Modular fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ to manage reusable and parametrized test resources efficiently.
*   **Unittest Compatibility:** Seamlessly run `unittest <https://docs.pytest.org/en/stable/how_to/unittest.html>`_ (and trial) test suites directly within pytest.
*   **Broad Python Support:** Compatible with Python 3.9+ and PyPy3.
*   **Extensible Architecture:** Benefit from a rich plugin ecosystem with over 1300+ `external plugins <https://docs.pytest.org/en/latest/reference/plugin_list.html>`_ and a thriving community.

## Getting Started

Here's a simple example of a test:

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

To run it:

```bash
$ pytest
============================= test session starts =============================
collected 1 items

test_sample.py F

================================== FAILURES ===================================
_________________________________ test_answer _________________________________

    def test_answer():
        assert inc(3) == 5
    E       assert 4 == 5
    E        +  where 4 = inc(3)

test_sample.py:5: AssertionError
========================== 1 failed in 0.04 seconds ===========================
```

## Documentation

Comprehensive documentation, including installation guides, tutorials, and more, can be found at [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/).

## Contributing and Support

*   **Bug Reports and Feature Requests:** Please use the `GitHub issue tracker <https://github.com/pytest-dev/pytest/issues>`_.
*   **Changelog:** Review the `Changelog <https://docs.pytest.org/en/stable/changelog.html>`_ for updates.
*   **Support the Project:** Consider supporting pytest through [Open Collective](https://opencollective.com/pytest).

## pytest for Enterprise

pytest is available as part of the Tidelift Subscription, offering commercial support and maintenance for the dependencies you use.  [Learn more.](https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## Security

To report any security vulnerabilities, please use the `Tidelift security contact <https://tidelift.com/security>`_.

## License

pytest is distributed under the `MIT`_ license and is free and open-source software.