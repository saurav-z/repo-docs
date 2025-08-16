# pytest-asyncio: Seamlessly Test Async Python Code with pytest

Easily test your asynchronous Python code with pytest-asyncio, a powerful pytest plugin designed for asyncio testing.

[![PyPI Version](https://img.shields.io/pypi/v/pytest-asyncio.svg)](https://pypi.python.org/pypi/pytest-asyncio)
[![CI Status](https://github.com/pytest-dev/pytest-asyncio/workflows/CI/badge.svg)](https://github.com/pytest-dev/pytest-asyncio/actions?workflow=CI)
[![Codecov](https://codecov.io/gh/pytest-dev/pytest-asyncio/branch/main/graph/badge.svg)](https://codecov.io/gh/pytest-dev/pytest-asyncio)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/pytest-asyncio.svg)](https://github.com/pytest-dev/pytest-asyncio)
[![Matrix Chat](https://img.shields.io/badge/Matrix-%23pytest--asyncio-brightgreen)](https://matrix.to/#/#pytest-asyncio:matrix.org)

**pytest-asyncio** is a pytest plugin that makes it easy to test code using the `asyncio` library.  This plugin allows you to write and run tests for your asynchronous code within the familiar pytest framework.  For more in-depth information, see the [official documentation](https://pytest-asyncio.readthedocs.io/en/latest/).

## Key Features

*   **Async Function Support:**  Write test functions using `async` and `await` syntax directly.
*   **Integration with pytest:**  Leverages the power and flexibility of the pytest testing framework.
*   **Simplified Async Testing:**  Streamlines the process of testing asynchronous code, making it easier to write and maintain tests.

## Example Usage

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

pytest will automatically detect and use the plugin after installation.

##  Important Notes

*   **unittest Integration:** `pytest-asyncio` does not support `unittest` subclasses.  Consider using `unittest.IsolatedAsyncioTestCase` or an async testing framework like `asynctest` if you are using the `unittest` library.

## Contributing

Contributions are welcome! Run tests with `tox` and ensure test coverage remains the same or improves before submitting a pull request.

## License

pytest-asyncio is licensed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).

For the original source code and further details, visit the [pytest-asyncio GitHub repository](https://github.com/pytest-dev/pytest-asyncio).