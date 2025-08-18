# Webassets: Streamline Your Web Development with Asset Management

**Webassets is a powerful Python library that simplifies asset management, enabling you to merge and compress your JavaScript and CSS files for optimal website performance.**

[View the Webassets project on GitHub](https://github.com/miracle2k/webassets)

## Key Features of Webassets:

*   **Asset Merging:** Combines multiple JavaScript and CSS files into fewer files to reduce HTTP requests.
*   **Asset Compression:** Minifies and compresses your assets (JavaScript, CSS, etc.) to reduce file sizes and improve loading times.
*   **Efficient Workflow:** Easily integrates into your Python web development workflow.
*   **Well-Tested & Reliable:**  The project is thoroughly tested.

## Installation

Install the latest development version using pip:

```bash
pip install webassets==dev
```

## Development Setup

To contribute or develop Webassets, follow these steps:

1.  **Set up your virtual environment:**

    ```bash
    uv venv
    ```

2.  **Install Python dependencies:**

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

    *   **Note:** Running tests requires Java 7 or later (needed for the Google Closure filter).

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Staying Up-to-Date

Since releases aren't on a strict schedule, it's recommended to use the latest code.  The build status badge (above) indicates the health of the project.