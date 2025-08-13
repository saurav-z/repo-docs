# Webassets: Streamline Your Web Development with Powerful Asset Management

**Webassets simplifies web development by efficiently merging and compressing your JavaScript and CSS files.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Asset Merging:** Combine multiple JavaScript and CSS files into a single, optimized file.
*   **Asset Compression:** Reduce file sizes using advanced compression techniques for faster loading times.
*   **Easy Integration:** Seamlessly integrate Webassets into your Python web development projects.
*   **Well-Tested:** Benefit from a robust and reliable asset management solution.

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Installation

Install the latest development version using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to the development of Webassets, follow these steps:

1.  **Set up a virtual environment:**
    ```bash
    uv venv
    ```
2.  **Install Python requirements:**
    ```bash
    uv pip install -r uv.lock
    ```
3.  **Install other requirements:**
    ```bash
    ./requirements-dev.sh
    ```
4.  **Run the tests:**
    ```bash
    ./run_tests.sh
    ```

**Note:** You will need Java 7 or later installed to run all tests, as it's required for the Google Closure filter.

## Development Status

[![CI Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)