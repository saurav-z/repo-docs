# webassets: Streamline Your Python Web Project's Assets

**webassets** is a powerful Python library that simplifies asset management, making it easy to merge and compress your JavaScript and CSS files for faster web performance.

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features:

*   **Asset Merging:** Combines multiple JavaScript and CSS files into single files, reducing HTTP requests.
*   **Compression:** Minifies your code (CSS and JavaScript) to reduce file sizes and improve loading times.
*   **Flexible Integration:** Integrates seamlessly with various Python web frameworks.
*   **Filter Support:**  Supports a range of filters for advanced processing, including Google Closure Compiler (requires Java).

## Installation

Install the latest development version using pip:

```bash
pip install webassets==dev
```

## Development

For development purposes, to run the tests, you'll need Java 7+ installed (required for the Google Closure filter).

**Setup:**

1.  Create a virtual environment using `uv venv`.
2.  Install Python requirements:

    ```bash
    uv pip install -r uv.lock
    ```

3.  Install other development requirements:

    ```bash
    ./requirements-dev.sh
    ```

**Testing:**

Run the tests using:

```bash
./run_tests.sh
```

## Documentation

Find detailed documentation at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Keeping Up-to-Date

Since releases are not on a regular schedule, the latest code is recommended.  The build status badge (above) indicates the build's health.

---