# pytest-asyncio: Effortlessly Test Asynchronous Python Code

**pytest-asyncio** seamlessly integrates the power of `asyncio` with `pytest`, enabling you to write robust and efficient tests for your asynchronous Python applications.  [Learn more on the original GitHub repo](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Asyncio Coroutine Support:** Directly use coroutines as test functions, allowing you to `await` code within your tests.
*   **Easy Integration:** A simple `pip install pytest-asyncio` is all it takes to get started.
*   **pytest Compatibility:**  Leverages the familiar and powerful pytest testing framework.
*   **Comprehensive Documentation:** Detailed documentation is available for in-depth understanding and usage:  [Documentation](https://pytest-asyncio.readthedocs.io/en/latest/).

## Example

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

Install `pytest-asyncio` using pip:

```bash
pip install pytest-asyncio
```

## Contributing

Contributions are highly encouraged!  Ensure test coverage remains consistent before submitting pull requests. Tests can be run with `tox`.

## License

pytest-asyncio is licensed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).