# pytest-asyncio: Seamlessly Test Asynchronous Python Code with pytest

Tired of wrestling with asynchronous code testing? **pytest-asyncio** is the perfect pytest plugin to effortlessly test your asyncio-based Python applications. Find the original project [here](https://github.com/pytest-dev/pytest-asyncio).

## Key Features

*   **Asynchronous Test Functions:** Directly use coroutines as test functions.
*   **`await` Support:** Easily `await` asynchronous code within your tests.
*   **pytest Integration:** Seamlessly integrates with the powerful pytest testing framework.
*   **Simple Installation:**  Install with a simple `pip install pytest-asyncio` command.

## How it Works

pytest-asyncio extends pytest to understand and execute asynchronous test functions.  Simply decorate your asynchronous test functions with `@pytest.mark.asyncio` and you can use `await` within your tests as needed.  

**Example:**

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_some_asyncio_code():
    # Assuming 'library.do_something()' is an async function
    res = await library.do_something()
    assert b"expected result" == res
```

## Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

pytest will automatically detect and use the plugin.

## Contributing

Contributions are highly encouraged!  Use `tox` to run the tests and ensure coverage remains the same or improves before submitting pull requests.

## Documentation

For comprehensive usage details and advanced features, please refer to the full [documentation](https://pytest-asyncio.readthedocs.io/en/latest/).

## License

pytest-asyncio is licensed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).