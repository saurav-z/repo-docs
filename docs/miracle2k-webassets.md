# Webassets: Streamline Your Python Web Project's Asset Management

**Webassets empowers Python developers to efficiently manage, merge, and compress their JavaScript and CSS files for optimal web performance.**  For detailed information, visit the [original repository on GitHub](https://github.com/miracle2k/webassets).

## Key Features

*   **Asset Merging:** Combine multiple CSS and JavaScript files into fewer HTTP requests.
*   **Asset Compression:** Reduce file sizes through minification and compression techniques (e.g., using Google Closure Compiler).
*   **Flexible Integration:** Seamlessly integrates with various Python web frameworks.
*   **Optimized Performance:** Improves website loading speeds and user experience.

## Installation

You can install the development version using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to Webassets, follow these steps:

1.  **Set up a virtual environment:**

    ```bash
    uv venv
    ```

2.  **Install Python requirements:**

    ```bash
    uv pip install -r uv.lock
    ```

3.  **Install other development requirements:**

    ```bash
    ./requirements-dev.sh
    ```

4.  **Run tests:**  Requires Java 7 or later (needed for the Google Closure filter).

    ```bash
    ./run_tests.sh
    ```

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Build Status

[![Build Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)