# webassets: Streamline Your Python Web Project's Asset Management

**webassets simplifies and optimizes your Python web development workflow by efficiently merging and compressing your JavaScript and CSS files.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Merge Assets:** Combine multiple JavaScript and CSS files into a single, optimized file.
*   **Compress Assets:** Reduce file sizes through compression, improving website loading speeds.
*   **Easy Integration:** Seamlessly integrates into your existing Python web projects.
*   **Widely Tested:** Built on a solid foundation with comprehensive testing, ensuring stability.

## Documentation

Comprehensive documentation is available to guide you through installation, setup, and usage:

*   [Read the Documentation](https://webassets.readthedocs.io/)

## Installation

You can install the development version using pip:

```bash
pip install webassets==dev
```

## Development

If you're contributing to webassets, you'll need Java 7+ installed for certain tests.

1.  **Set up a virtual environment and install Python requirements:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install additional development requirements:**

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run the tests:**

    ```bash
    ./run_tests.sh
    ```