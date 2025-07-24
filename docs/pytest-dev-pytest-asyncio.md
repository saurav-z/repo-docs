# pytest-asyncio: Seamlessly Test Async Python Code

Easily test your asynchronous Python code with pytest-asyncio, a powerful plugin that integrates asyncio support directly into your pytest testing environment.  [See the original repo](https://github.com/pytest-dev/pytest-asyncio) for more information.

## Key Features

*   **Async Function Support:**  Write pytest tests using `async` functions (coroutines).
*   **Await in Tests:**  Use `await` within your tests to interact with asynchronous code.
*   **Simple Installation:**  Installation is a breeze with a single pip command.
*   **pytest Integration:** Leverages the familiar pytest framework for test discovery and execution.
*   **Comprehensive Documentation:** Detailed documentation is available to guide you.

## How it Works

pytest-asyncio allows you to easily test code that leverages the `asyncio` library.  By using the `@pytest.mark.asyncio` marker, you can designate async test functions that `pytest` will execute correctly.

```python
import pytest

@pytest.mark.asyncio
async def test_my_async_function():
    result = await some_async_function()
    assert result == "expected result"
```

## Installation

To install pytest-asyncio:

```bash
pip install pytest-asyncio
```

## Contributing

Contributions are welcome!  Ensure tests pass and maintain code coverage before submitting pull requests. Run tests with `tox`.

## License

pytest-asyncio is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).