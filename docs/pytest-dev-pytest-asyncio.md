# pytest-asyncio: Seamlessly Test Asynchronous Python Code

Easily test your asynchronous Python code with `pytest-asyncio`, a powerful pytest plugin that simplifies testing asyncio applications. You can find the original source code and more information at [the pytest-asyncio GitHub repository](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Async Function Support:** Write tests using `async` and `await` directly in your test functions.
*   **Easy Integration:**  Simply install the plugin and start writing async tests – no complex setup required.
*   **pytest Compatibility:**  Leverages the familiar and flexible pytest testing framework.
*   **Coroutine as Test Functions:** Enables the use of coroutines as test functions.
*   **Clear Assertions:** Use standard `assert` statements within your async tests.

## Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

pytest will automatically detect and use the plugin after installation.

## Example Usage

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

##  Important Note:

pytest-asyncio does not support test classes subclassing the standard `unittest` library.  Consider using `unittest.IsolatedAsyncioTestCase` or an async framework like `asynctest` for these scenarios.

## Contributing

Contributions are welcome! Ensure test coverage remains consistent before submitting a pull request.  Use `tox` to run tests.

## License

pytest-asyncio is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).

## Additional Resources

*   [Documentation](https://pytest-asyncio.readthedocs.io/en/latest/)
*   [GitHub Repository](https://github.com/pytest-dev/pytest-asyncio)