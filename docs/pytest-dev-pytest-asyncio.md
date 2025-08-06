# pytest-asyncio: Effortlessly Test Asynchronous Python Code

**pytest-asyncio** is a powerful pytest plugin that simplifies the testing of asynchronous Python code, allowing you to seamlessly integrate `asyncio` into your testing workflow.  You can find the original project on GitHub: [pytest-dev/pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Async Test Function Support:** Write and execute tests using coroutines directly, enabling the use of `await` within your test functions.
*   **Easy Integration:** Seamlessly integrates with your existing pytest setup.
*   **Simple Installation:** Install with a single pip command.
*   **Comprehensive Documentation:** Detailed documentation available to get you started quickly.

## How it Works

pytest-asyncio enables you to write asynchronous tests with ease. Simply decorate your test functions with `@pytest.mark.asyncio` and use `async` and `await` as you would in your application code:

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

## Contributing

Contributions are welcome! Run tests with `tox` and ensure test coverage remains consistent before submitting pull requests.

## License

pytest-asyncio is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).