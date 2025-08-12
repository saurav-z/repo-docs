# pytest-asyncio: Easily Test Asynchronous Python Code

Tired of struggling to test your asynchronous Python code? **pytest-asyncio makes it easy to write and run tests for code that uses the `asyncio` library, seamlessly integrating with the popular `pytest` testing framework.**

[Check out the original repository](https://github.com/pytest-dev/pytest-asyncio)

## Key Features

*   **Asyncio Support:** Enables testing of code that uses the `asyncio` library, including coroutines.
*   **Await in Tests:** Allows you to `await` code directly within your test functions.
*   **Simple Integration:**  A pytest plugin that is automatically discovered and loaded when installed.

## How it Works

pytest-asyncio provides support for coroutines as test functions. This allows you to write tests like this:

```python
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

pytest will automatically discover and use the plugin after installation.

## Contributing

Contributions are welcome!  Ensure test coverage remains consistent or improves before submitting a pull request.  Use `tox` to run the tests.

## License

pytest-asyncio is available under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).