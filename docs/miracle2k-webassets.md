# Webassets: Streamline Your Web Project's Assets

**Webassets is a powerful Python library designed to simplify the management, merging, and compression of your JavaScript and CSS files, leading to faster loading times and a better user experience.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Asset Management:** Efficiently organize and manage your static assets.
*   **Merging:** Combine multiple JavaScript and CSS files into single files to reduce HTTP requests.
*   **Compression:** Minify and compress your assets to reduce file sizes and improve loading speed.
*   **Filter Support:**  Supports filters for further processing, including Google Closure Compiler.
*   **Flexible Integration:** Designed for seamless integration with Python web development projects.

## Installation

You can install the latest development version using pip:

```bash
pip install webassets==dev
```

*(Note: Releases may not occur on a regular schedule, but the code is well-tested. Ensure the build status badge on GitHub is green for stability.)*

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Development

### Prerequisites

*   Python 3.x
*   Java 7 (or later) is required to run the Google Closure filter.

### Setup

1.  Create a virtual environment using `uv`:

    ```bash
    uv venv
    ```

2.  Install Python requirements:

    ```bash
    uv pip install -r uv.lock
    ```

3.  Install other development requirements:

    ```bash
    ./requirements-dev.sh
    ```

4.  Run tests:

    ```bash
    ./run_tests.sh
    ```