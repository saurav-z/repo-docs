# pytest-asyncio: Easily Test Asynchronous Python Code

**pytest-asyncio** is a powerful pytest plugin that simplifies testing asynchronous code, making it easier than ever to ensure the reliability of your async applications.  You can find the original repository here: [pytest-asyncio on GitHub](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Coroutine Support:**  Write tests that directly use coroutines, allowing you to `await` asynchronous code within your tests.
*   **Seamless Integration:**  Integrates flawlessly with pytest, making it easy to incorporate asynchronous testing into your existing test suite.
*   **Simple Usage:**  Simply mark your test functions with `@pytest.mark.asyncio` to enable asynchronous testing.
*   **Comprehensive Documentation:** Detailed documentation is available to guide you through installation and usage.

## Example

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

Installing pytest-asyncio is straightforward:

```bash
pip install pytest-asyncio
```

pytest will automatically detect and use the plugin.

## Contributing

Contributions are highly encouraged!  Ensure test coverage remains consistent before submitting pull requests. Use `tox` to run tests.

## License

pytest-asyncio is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).