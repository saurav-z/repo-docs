# Webassets: Simplify Your Python Web Development with Powerful Asset Management

**Webassets is your go-to solution for streamlining asset management in Python web projects, making it easy to merge and compress your JavaScript and CSS files for optimized performance.**

[View the project on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Asset Bundling:** Efficiently merge multiple JavaScript and CSS files into single bundles.
*   **Compression:** Minify and compress your assets (e.g., JavaScript, CSS) to reduce file sizes and improve page load times.
*   **Flexible Integration:** Seamlessly integrates with various Python web frameworks.

## Installation

Install the development version using pip:

```bash
pip install webassets==dev
```

or download a tarball: [download a tarball](http://github.com/miracle2k/webassets/tarball/master#egg=webassets-dev)

## Development

For development, you'll need Python and optionally Java 7 (required for certain filters like Google Closure).

1.  **Set up a virtual environment using `uv`:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install additional development requirements:**

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run Tests:**

    ```bash
    ./run_tests.sh
    ```

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

**Note:** Since releases are not on a fixed schedule, consider using the latest code from the repository. The build status is a good indicator of stability.

## Continuous Integration

[![Build Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)