# pytest-asyncio: Seamlessly Test Asynchronous Python Code

Easily test your asynchronous Python code with `pytest-asyncio`, a powerful plugin for the popular `pytest` testing framework.

[Explore the original repository on GitHub](https://github.com/pytest-dev/pytest-asyncio)

## Key Features:

*   **Asyncio Test Function Support:** Enables direct use of coroutines as test functions, allowing you to easily `await` asynchronous code within your tests.
*   **Simple Integration:**  Install `pytest-asyncio` with a simple `pip install` command, and it's ready to go – pytest will automatically detect and use it.
*   **Flexible Testing:**  Provides a flexible environment for testing asynchronous code.

## How it Works:

`pytest-asyncio` extends `pytest` to natively support testing code that leverages the `asyncio` library.  Decorate your test functions with `@pytest.mark.asyncio` and write tests that `await` the results of asynchronous functions:

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

To get started with `pytest-asyncio`:

```bash
pip install pytest-asyncio
```

## Documentation

For more in-depth information and examples, consult the full documentation: [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/en/latest/)

## Contributing

We welcome contributions!  Run tests using `tox` and ensure coverage remains consistent before submitting pull requests.