# pytest-asyncio: Easily Test Asynchronous Python Code with pytest

Tired of struggling to test your asynchronous Python code? **pytest-asyncio simplifies testing asyncio code, allowing you to effortlessly write and run asynchronous tests with pytest.** [See the original repository](https://github.com/pytest-dev/pytest-asyncio) for more details.

## Key Features

*   **Asynchronous Test Function Support:** Write test functions using `async def` and easily `await` within your tests.
*   **Seamless Integration with pytest:**  Works as a standard pytest plugin, requiring no complex setup.
*   **Simple Installation:** Install with a single pip command.
*   **Comprehensive Documentation:** Detailed documentation available for guidance.

## Getting Started

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

Now, simply use the `@pytest.mark.asyncio` decorator to mark your asynchronous test functions:

```python
import pytest

@pytest.mark.asyncio
async def test_my_async_function():
    result = await my_async_function()
    assert result == "expected_value"
```

## Important Notes

*   pytest-asyncio supports coroutines as test functions.
*   Test classes subclassing `unittest` are not directly supported. Consider using `unittest.IsolatedAsyncioTestCase` or an async testing framework such as `asynctest`.

## Contributing

Contributions are highly encouraged!  Run tests using `tox` and ensure test coverage remains consistent before submitting a pull request.

## License

pytest-asyncio is licensed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).