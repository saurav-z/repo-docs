# pytest-asyncio: Effortlessly Test Asynchronous Python Code

**Enhance your Python testing workflow with pytest-asyncio, a powerful pytest plugin designed to simplify testing asynchronous code.**

[View the original repository on GitHub](https://github.com/pytest-dev/pytest-asyncio)

## Key Features

*   **Seamless Integration with `asyncio`:** Enables direct testing of code utilizing the `asyncio` library.
*   **Coroutine Support as Test Functions:** Allows you to define and execute coroutines directly within your pytest tests.
*   **`await` Keyword Usage:**  Easily `await` asynchronous operations within your tests for cleaner, more readable code.
*   **Simple Installation:**  Install with a single `pip` command, and pytest automatically detects and utilizes the plugin.

## How It Works

pytest-asyncio extends pytest to handle asynchronous test functions. This means you can use the `@pytest.mark.asyncio` decorator to mark a function as an asynchronous test, and then use the `await` keyword to interact with asynchronous code within the test.

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_some_asyncio_code():
    # Assume 'library.do_something()' is an async function
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

```bash
pip install pytest-asyncio
```

## Contributing

Contributions are greatly appreciated!  Run tests with `tox` and ensure test coverage remains consistent before submitting a pull request.

## License

pytest-asyncio is distributed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).

## Further Information

*   [Read the full documentation](https://pytest-asyncio.readthedocs.io/en/latest/)