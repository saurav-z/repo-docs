# Webassets: Streamline Your Python Web Development with Efficient Asset Management

**Webassets** is a powerful asset management tool for Python web developers, simplifying the merging and compression of your JavaScript and CSS files for optimal performance. [View the original repository on GitHub](https://github.com/miracle2k/webassets).

## Key Features:

*   **Merge and Compress Assets:** Combine and minimize your JavaScript and CSS files to reduce HTTP requests and improve page load times.
*   **Easy Integration:** Seamlessly integrate Webassets into your existing Python web projects.
*   **Filter Support:** Includes support for a wide range of filters, like Google Closure, to optimize your assets.
*   **Development-Friendly:** Well-tested and actively maintained, allowing you to use the latest code.

## Installation

Install the development version using pip:

```bash
pip install webassets==dev
```

## Documentation

Comprehensive documentation is available at:

*   [Webassets Documentation](https://webassets.readthedocs.io/)

## Development

To contribute to Webassets, follow these steps:

1.  **Set up your Python environment using `uv`:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```
2.  **Install other requirements:**

    ```bash
    ./requirements-dev.sh
    ```
3.  **Run tests:**

    ```bash
    ./run_tests.sh
    ```

**Note:** You will need Java 7 or later installed to run all tests (required for filters such as Google Closure).

## Further Resources:

*   **Download:** [Development Version Tarball](http://github.com/miracle2k/webassets/tarball/master#egg=webassets-dev)
*   **CI Status:** [![CI Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)