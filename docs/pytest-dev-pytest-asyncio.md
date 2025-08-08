# pytest-asyncio: Seamlessly Test Your Asynchronous Python Code

**Easily test your asynchronous Python code with pytest-asyncio, a powerful pytest plugin that brings the full power of pytest to asyncio.** Learn more and contribute on the original repository: [https://github.com/pytest-dev/pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)

## Key Features of pytest-asyncio:

*   **Enables Testing of Coroutines:** Write tests using `async` and `await` directly within your pytest test functions.
*   **pytest Integration:** Seamlessly integrates with the popular pytest testing framework.
*   **Simple Installation:** Install with a single `pip` command.
*   **Comprehensive Documentation:** Detailed documentation available at [https://pytest-asyncio.readthedocs.io/en/latest/](https://pytest-asyncio.readthedocs.io/en/latest/).
*   **Open Source:**  Available under the Apache License 2.0.

## Getting Started

### Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

## Usage

Simply mark your test functions with `@pytest.mark.asyncio` to enable async support:

```python
import pytest

@pytest.mark.asyncio
async def test_my_async_function():
    result = await my_async_function()
    assert result == "expected"
```

## Contributing

Contributions are highly encouraged!  Run tests with `tox` and ensure coverage remains consistent before submitting pull requests.