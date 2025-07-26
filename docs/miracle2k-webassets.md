# Webassets: Streamline Your Web Development Asset Management

Webassets is a powerful Python library designed to efficiently merge and compress your JavaScript and CSS files, optimizing your web projects.  **(Link to original repo: [https://github.com/miracle2k/webassets](https://github.com/miracle2k/webassets))**

## Key Features

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single, optimized files.
*   **Compression:**  Reduce file sizes using various compression techniques (e.g., JavaScript minification, CSS compression).
*   **Easy Integration:**  Integrates seamlessly into Python web development workflows.
*   **Well-Tested:** Reliably handles your asset management needs, as indicated by the build status badge (see below).

## Installation

Install the latest development version using pip:

```bash
pip install webassets==dev
```

## Development

This section provides instructions for contributing to Webassets development.

**Prerequisites:**

*   Python 3.x
*   Java 7 or higher (required for the Google Closure filter)

**Steps:**

1.  **Set up a virtual environment using uv:**

    ```bash
    uv venv
    ```

2.  **Install Python requirements using uv:**

    ```bash
    uv pip install -r uv.lock
    ```

3.  **Install other requirements:**

    ```bash
    ./requirements-dev.sh
    ```

4.  **Run tests:**

    ```bash
    ./run_tests.sh
    ```

## Documentation & Resources

*   **Documentation:** [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)
*   **Development Version Tarball:**  [Download a tarball](http://github.com/miracle2k/webassets/tarball/master#egg=webassets-dev)
*   **Build Status:** [![Build Status](https://github.com/miracle2k/webassets/actions/workflows/ci.yml/badge.svg)](https://github.com/miracle2k/webassets/actions/workflows/ci.yml)
*   **Google Closure:** [Google Closure Compiler](https://github.com/google/closure-compiler/wiki/FAQ#the-compiler-crashes-with-unsupportedclassversionerror-or-unsupported-majorminor-version-510)