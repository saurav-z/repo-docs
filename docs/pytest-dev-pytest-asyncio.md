# pytest-asyncio: Seamlessly Test Your Async Python Code with pytest

Enhance your Python testing workflow by easily testing asynchronous code with `pytest-asyncio`, a powerful plugin for the popular `pytest` framework. Check out the original repository [here](https://github.com/pytest-dev/pytest-asyncio) for more details.

## Key Features of pytest-asyncio:

*   **Asyncio Support:** Enables the use of `asyncio` features directly within your `pytest` tests.
*   **Coroutine Test Functions:** Allows you to define test functions as coroutines, using `async` and `await`.
*   **Simple Integration:** Integrates seamlessly with your existing `pytest` setup.
*   **Easy Installation:** Installs quickly via pip: `pip install pytest-asyncio`.
*   **Comprehensive Documentation:** Detailed documentation available [here](https://pytest-asyncio.readthedocs.io/en/latest/).

## How it Works:

pytest-asyncio allows you to write tests like this:

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

This allows you to test async code directly within your tests.

## Installation

To get started, simply install pytest-asyncio:

```bash
pip install pytest-asyncio
```

## Contributing

Contributions are highly encouraged!  Ensure test coverage remains consistent before submitting a pull request.  Run tests using `tox`.

## License

pytest-asyncio is licensed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).