# Webassets: Streamline Your Python Web Project's Assets

**Webassets is a powerful Python package that simplifies asset management for your web projects by merging and compressing your JavaScript and CSS files, leading to faster loading times and improved performance.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features of Webassets:

*   **Asset Merging:** Combine multiple JavaScript and CSS files into single files.
*   **Asset Compression:** Minify your JavaScript and CSS code to reduce file sizes.
*   **Flexible Integration:** Easy to integrate into various Python web frameworks.

## Documentation

Comprehensive documentation is available to guide you through the setup and use of Webassets:

*   [Webassets Documentation](https://webassets.readthedocs.io/)

## Installation

While releases aren't always on a fixed schedule, the project is well-tested and the latest code is generally stable.  To install, use pip:

```bash
pip install webassets==dev
```

## Development

To contribute to the development of Webassets, follow these steps:

1.  **Set up a virtual environment using uv:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install development requirements:**

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run the tests:**

    ```bash
    ./run_tests.sh
    ```

    **Note:**  Running tests requires Java 7 or higher to be installed, as it's needed for the Google Closure filter.