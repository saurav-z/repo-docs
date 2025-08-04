# pytest-asyncio: Seamlessly Test Asynchronous Python Code with pytest

Easily test your asynchronous Python code with `pytest-asyncio`, a powerful plugin that integrates `asyncio` support directly into your `pytest` testing framework.  Check out the original repo [here](https://github.com/pytest-dev/pytest-asyncio).

## Key Features of pytest-asyncio

*   **Asyncio Coroutine Support:** Write test functions as coroutines using `async` and `await` to test asynchronous code.
*   **pytest Integration:**  Works seamlessly with your existing pytest setup.
*   **Simple Installation:** Install with a single `pip` command.
*   **Test Asynchronous Code:** Easily test code that utilizes the `asyncio` library.
*   **Comprehensive Documentation:**  Detailed documentation to guide you through the features and usage.

## Getting Started

### Installation

Install `pytest-asyncio` using pip:

```bash
pip install pytest-asyncio
```

pytest will automatically discover and use the plugin.

### Example

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Important Considerations

*   `pytest-asyncio` supports coroutines as test functions, allowing the use of `await` inside tests.
*   Test classes subclassing the standard `unittest` library are **not** supported.  Use `unittest.IsolatedAsyncioTestCase` or an async framework like `asynctest` instead for class-based async tests.

## Contributing

Contributions are highly encouraged! Run tests with `tox` and ensure that code coverage remains the same or increases before submitting a pull request.

## License

`pytest-asyncio` is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).