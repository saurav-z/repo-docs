# Webassets: Streamline Your Python Web Project's Assets with Powerful Management

**Webassets is the ultimate asset management tool for Python web developers, helping you merge and compress your JavaScript and CSS files to optimize performance.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Merge and Compress:** Efficiently combines and minimizes your JavaScript and CSS files for faster loading times.
*   **Easy Integration:** Seamlessly integrates with your existing Python web development workflow.
*   **Well-Tested:** Relies on a stable and reliable codebase.
*   **Development Version Available:** Access the latest features by installing the development version.

## Installation

You can install webassets using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to webassets, follow these steps:

1.  **Install Python requirements using `uv`:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```
2.  **Install other requirements:**

    ```bash
    ./requirements-dev.sh
    ```
3.  **Run the tests:**

    ```bash
    ./run_tests.sh
    ```

## Documentation

Detailed documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Important Notes
*   For development, Java 7 or higher is required (needed for filters like Google Closure).
*   The build status can be checked in the badge above.