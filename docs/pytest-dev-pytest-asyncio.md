# pytest-asyncio: Seamlessly Test Your Async Python Code

**Enhance your Python testing with pytest-asyncio, a powerful pytest plugin designed to effortlessly test code using the `asyncio` library.**

[View the original repository on GitHub](https://github.com/pytest-dev/pytest-asyncio)

## Key Features

*   **Asyncio Support:** Enables testing of code that utilizes the `asyncio` library.
*   **Coroutine Test Functions:** Allows you to define coroutines directly as test functions.
*   **Await Inside Tests:**  Easily `await` async functions and operations within your tests, streamlining your testing workflow.
*   **Integration with pytest:**  Fully integrates with the popular pytest testing framework.
*   **Simple Installation:** Install with a single `pip install` command.

## How it Works

pytest-asyncio seamlessly integrates with pytest, allowing you to write tests that execute async code.  By using the `@pytest.mark.asyncio` marker, you can designate an async function as a test and then `await` calls within that function.

```python
import pytest

@pytest.mark.asyncio
async def test_my_async_function():
    result = await some_async_function()
    assert result == "expected"
```

## Installation

To install pytest-asyncio, simply run:

```bash
pip install pytest-asyncio
```

pytest will automatically discover and utilize the plugin after installation.

## Contributing

Contributions are encouraged! Run tests with `tox` and ensure that test coverage remains consistent before submitting a pull request.

## Documentation

For in-depth information, including detailed usage examples and advanced features, refer to the comprehensive documentation: [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/en/latest/)

## License

pytest-asyncio is released under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).