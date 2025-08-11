# Webassets: Simplify Your Python Web Project's Asset Management

**Streamline your web development workflow and boost performance with Webassets, the powerful Python asset management tool.**

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features

*   **Merge and Compress:** Efficiently combines and minifies your JavaScript and CSS files to reduce HTTP requests and improve page load times.
*   **Python-Based:** Integrates seamlessly with your Python web development projects.
*   **Flexible:** Provides a range of filters and options for advanced asset processing.

## Installation

Install Webassets using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to Webassets, follow these steps:

1.  **Set up a virtual environment using uv:**

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

4.  **Run tests:**

    ```bash
    ./run_tests.sh
    ```

    *Note: You will need Java 7 or higher installed to run all tests (required for the Google Closure filter).*

## Documentation

For comprehensive information and usage examples, refer to the official documentation:

*   [Webassets Documentation](https://webassets.readthedocs.io/)

## Staying Up-to-Date

Since releases are not on a fixed schedule, it's recommended to use the latest code. Webassets is well-tested, and the build status badge above provides a good indication of stability. You can also download a tarball of the development version.