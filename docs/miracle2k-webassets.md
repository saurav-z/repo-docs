# Webassets: Streamline Your Python Web Project's Assets

**Webassets simplifies asset management in your Python web projects, making it easy to merge and compress your JavaScript and CSS files for improved performance and a better user experience.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features:

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files.
*   **Compression:** Minify and compress your assets to reduce file sizes.
*   **Easy Integration:** Designed for straightforward integration into your Python web development workflow.

## Installation

To install Webassets, use pip:

```bash
pip install webassets==dev
```

## Development

For development, you'll need Python requirements and potentially Java 7 (for certain filters like Google Closure).

**Steps:**

1.  **Create a virtual environment and install Python requirements:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install other development requirements:**

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run the tests:**

    ```bash
    ./run_tests.sh
    ```

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

**Note:**  While releases are not on a strict schedule, the project is well-tested.  Check the build status icon (above) for a reassuring green to ensure stability when using the latest code.