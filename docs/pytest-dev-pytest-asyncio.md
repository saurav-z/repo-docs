# pytest-asyncio: Seamlessly Test Asynchronous Python Code

Easily test your asynchronous Python code with pytest-asyncio, a powerful plugin that brings the flexibility of pytest to your asyncio projects.  For the original project, visit the [pytest-asyncio GitHub repository](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Asyncio Test Function Support:**  Write tests using async functions (coroutines) that you can `await` directly within your test code.
*   **Integration with pytest:** Leverages the familiar and robust features of pytest, including fixtures, markers, and more.
*   **Simplified Asynchronous Testing:**  Makes it easy to test asynchronous code with a clean and concise syntax.

## Installation

Get started with pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

pytest will automatically discover and use the plugin after installation.

## Usage Example

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()  # Assuming 'library' is an async library
    assert b"expected result" == res
```

## Important Notes

*   pytest-asyncio is designed to work directly with pytest.
*   Test classes subclassing the standard `unittest` library are not directly supported.  Consider using `unittest.IsolatedAsyncioTestCase` or a dedicated async testing framework like `asynctest` for those scenarios.

## Contributing

Contributions are highly encouraged!  Ensure your code passes tests using `tox` and maintains or improves test coverage before submitting pull requests.

## License

pytest-asyncio is released under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).