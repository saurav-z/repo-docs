# Webassets: Streamline Your Python Web Development with Asset Management

**Effortlessly merge and compress your JavaScript and CSS files to optimize your web applications with Webassets.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files to reduce HTTP requests.
*   **Asset Compression:** Minimize file sizes using various compression techniques for faster loading times.
*   **Python-Based:** Seamlessly integrates with your Python web development workflow.
*   **Well-Tested:** Relies on a robust testing suite to ensure reliability.

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Installation

While releases are not on a regular schedule, the latest code is encouraged.  Install the latest development version:

```bash
pip install webassets==dev
```

## Development

### Requirements

*   Java 7 (required for tests using the Google Closure filter)

### Setup

1.  Install Python requirements using `uv`:

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  Install other development requirements:

    ```bash
    ./requirements-dev.sh
    ```

3.  Run the tests:

    ```bash
    ./run_tests.sh