# Webassets: Streamline Your Python Web Project's Assets

**Webassets simplifies asset management in your Python web projects by merging and compressing JavaScript and CSS files for optimal performance.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files to reduce HTTP requests.
*   **Compression:**  Compress your assets (JS and CSS) to minimize file sizes, improving page load times.
*   **Integration:** Designed for easy integration with Python web frameworks.
*   **Well-Tested:** Rely on a robust library with comprehensive testing.

## Installation

Install the development version using pip:

```bash
pip install webassets==dev
```

Or download a tarball: [https://github.com/miracle2k/webassets/tarball/master#egg=webassets-dev](https://github.com/miracle2k/webassets/tarball/master#egg=webassets-dev)

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Development

Follow these steps to contribute to the project:

1.  **Set up a virtual environment using uv**:

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install development requirements**:

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run tests**:

    ```bash
    ./run_tests.sh
    ```

    *Note: Running tests requires Java 7 or later for the Google Closure filter.*

[![Build Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)