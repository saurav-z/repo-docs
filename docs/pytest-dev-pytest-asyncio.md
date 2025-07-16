# pytest-asyncio: Seamlessly Test Asynchronous Python Code

Easily test your asynchronous Python code with pytest-asyncio, a powerful plugin that brings asyncio support to your pytest testing framework.  For more details, please visit the original repository [here](https://github.com/pytest-dev/pytest-asyncio).

## Key Features of pytest-asyncio

*   **Enables Async Test Functions:** Decorate your test functions with `@pytest.mark.asyncio` to write asynchronous tests using `async` and `await` syntax.
*   **Simplified Testing of Async Code:** Makes it straightforward to test code that utilizes the `asyncio` library.
*   **Easy Integration:**  Simply install the plugin and pytest will automatically discover and run your async tests.

## Getting Started

### Installation

Install pytest-asyncio using pip:

```bash
pip install pytest-asyncio
```

## Contributing

Contributions are highly encouraged! Ensure test coverage remains the same or improves before submitting a pull request.  Use `tox` to run the tests.

## Documentation

Detailed documentation is available at [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/en/latest/).

## License

pytest-asyncio is licensed under the [Apache License 2.0](https://github.com/pytest-dev/pytest-asyncio/blob/main/LICENSE).