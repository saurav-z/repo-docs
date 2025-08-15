# Webassets: Streamline Your Python Web Development with Powerful Asset Management

**Webassets simplifies your Python web development workflow by providing a robust asset management solution for merging and compressing your JavaScript and CSS files.**  Optimize your web application's performance and improve user experience with this versatile tool.

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features of Webassets:

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single, optimized files to reduce HTTP requests.
*   **Asset Compression:**  Compress your JavaScript and CSS files to minimize file sizes and improve page load times.
*   **Flexible Integration:** Easily integrate webassets with various Python web frameworks.
*   **Well-Tested & Reliable:** Benefit from a well-tested codebase, ensuring stability and consistent performance.

## Getting Started

### Installation

Install the development version using pip:

```bash
pip install webassets==dev
```

### Documentation

Comprehensive documentation is available at:  [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

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

## Build Status

[![Build Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)

**Note:** Development requires Java 7 or higher for certain filters (e.g., Google Closure).