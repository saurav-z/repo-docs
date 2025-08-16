# Webassets: Streamline Your Python Web Development with Powerful Asset Management

**Webassets** is a robust Python asset management application designed to simplify and optimize your web projects by merging and compressing your JavaScript and CSS files. For more details, visit the [Webassets GitHub repository](https://github.com/miracle2k/webassets).

## Key Features

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files for improved performance.
*   **Compression:** Reduce file sizes through compression (e.g., minification) to speed up page loading times.
*   **Easy Integration:** Seamlessly integrate with your existing Python web development workflow.
*   **Well-Tested:** Benefit from a stable and reliable asset management solution.

## Installation

You can install the latest development version using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to the development of Webassets, follow these steps:

1.  **Set up your development environment:**

    *   Create a virtual environment using `uv`:

        ```bash
        uv venv
        ```
    *   Install Python requirements:

        ```bash
        uv pip install -r uv.lock
        ```

    *   Install other development requirements:

        ```bash
        ./requirements-dev.sh
        ```

2.  **Run tests:**

    ```bash
    ./run_tests.sh
    ```

    *Note: Running tests requires Java 7 or higher for the Google Closure filter.*

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

---