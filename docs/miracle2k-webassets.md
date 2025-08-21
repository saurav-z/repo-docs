# Webassets: Streamline Your Python Web Development with Asset Management

**Webassets simplifies your Python web development workflow by merging and compressing your JavaScript and CSS files for optimized performance.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features:

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files to reduce HTTP requests.
*   **Asset Compression:** Minimize file sizes through compression, leading to faster page load times.
*   **Simplified Workflow:** Manage and optimize your static assets efficiently.
*   **Flexible Integration:** Easily integrates into your existing Python web projects.

## Installation

You can install the latest development version using pip:

```bash
pip install webassets==dev
```

## Documentation

Comprehensive documentation is available at:  [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Development

To contribute to the development of Webassets, follow these steps:

1.  **Set up a virtual environment**:
    ```bash
    uv venv
    ```
2.  **Install Python requirements:**
    ```bash
    uv pip install -r uv.lock
    ```
3.  **Install additional requirements:**
    ```bash
    ./requirements-dev.sh
    ```
4.  **Run tests:**
    ```bash
    ./run_tests.sh
    ```

**Note:** Running tests requires Java 7 or higher to be installed (required for some filters like Google Closure).

## Build Status

[![Build Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)