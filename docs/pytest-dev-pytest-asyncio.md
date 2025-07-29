# pytest-asyncio: Seamlessly Test Your Asynchronous Python Code

Effortlessly test your asynchronous Python applications with `pytest-asyncio`, a powerful plugin that extends the functionality of the popular pytest testing framework.  **[View the original repository on GitHub](https://github.com/pytest-dev/pytest-asyncio)**

## Key Features of pytest-asyncio:

*   **Enables Async Test Functions:** Write and execute tests using Python's `asyncio` library directly within your pytest setup.
*   **Awaits Coroutines:**  Easily `await` asynchronous code inside your test functions for clean and readable testing.
*   **Simple Installation:**  Install with a straightforward `pip install pytest-asyncio` command.
*   **pytest Integration:** Seamlessly integrates with the pytest testing framework, leveraging its existing features and extensibility.
*   **Comprehensive Documentation:**  Detailed documentation is available to guide you through the plugin's usage and advanced features.

## How pytest-asyncio Works

This plugin empowers you to write asynchronous tests with ease. For example:

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

pytest will automatically discover and utilize the plugin after installation.

## Contributing

Contributions are highly encouraged! Ensure test coverage remains consistent before submitting a pull request; run tests with `tox`.

## License

pytest-asyncio is released under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).