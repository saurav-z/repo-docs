# pytest-asyncio: Seamlessly Test Asynchronous Python Code with pytest

**Effortlessly test your asynchronous Python code using pytest with the pytest-asyncio plugin, enabling cleaner, more readable tests.**

[See the original repository on GitHub](https://github.com/pytest-dev/pytest-asyncio)

## Key Features of pytest-asyncio

*   **Asyncio Support:** Directly test code that uses the `asyncio` library.
*   **Coroutine Test Functions:** Write tests using coroutines (functions defined with `async def`).
*   **Awaitable Code:**  Use `await` within your tests to interact with asynchronous functions and operations.
*   **Easy Integration:** Seamlessly integrates with the pytest testing framework.

## How it Works

pytest-asyncio enhances pytest to recognize and execute coroutines as test functions.  Simply decorate your test functions with `@pytest.mark.asyncio` and `await` your asynchronous code within them.

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == "expected_value"
```

## Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

pytest will automatically discover and use the plugin.

## Contributing

Contributions are welcome! Run tests using `tox` and ensure that test coverage remains at least the same before submitting pull requests.

## Further Information

*   **Documentation:**  [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/en/latest/)
*   **License:** Apache License 2.0 ([LICENSE](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE))