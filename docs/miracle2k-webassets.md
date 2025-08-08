# Webassets: Powerful Asset Management for Python Web Development

**Webassets simplifies and optimizes your Python web projects by merging and compressing your JavaScript and CSS files.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features:

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files to reduce HTTP requests.
*   **Asset Compression:** Minify and compress your assets to improve page load times.
*   **Flexible:** Works seamlessly with various Python web frameworks.
*   **Well-Tested:** Benefit from a robust and reliable asset management solution.

## Installation

Install the latest development version using pip:

```bash
pip install webassets==dev
```

## Documentation

Detailed documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Development

For development, follow these steps:

1.  **Install Python requirements with uv:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install other requirements:**

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run the tests:**

    ```bash
    ./run_tests.sh
    ```

**Note:** Running tests requires Java 7 or higher for certain filters (e.g., Google Closure).

## Contributing

We encourage you to contribute to `webassets`!  Please refer to the documentation and existing issues for guidelines on how to contribute.