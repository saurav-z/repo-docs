# Test Asynchronous Code with pytest-asyncio

**Easily write and run tests for your Python asynchronous code using pytest-asyncio, a powerful pytest plugin.** Learn more and contribute on the [original repository](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Enables Testing of `asyncio` Code:** Write tests that directly interact with asynchronous code, simplifying the testing process.
*   **Coroutine Support:**  Use coroutines (async functions) as test functions, allowing you to `await` code directly within your tests.
*   **Simple Integration with pytest:**  pytest-asyncio is a pytest plugin, so installation and usage are straightforward.
*   **Comprehensive Documentation:** Extensive documentation is available to help you get started and explore advanced features.

## Example Usage

```python
import pytest
import library  # Assuming your async library

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

Get started with pytest-asyncio in seconds:

```bash
pip install pytest-asyncio
```

## License

pytest-asyncio is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).

## Contributing

We welcome contributions!  Run tests with `tox` and ensure coverage remains consistent before submitting a pull request.