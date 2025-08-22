# Webassets: Optimize Your Python Web App's Assets

**Webassets is the go-to Python asset management tool, helping you merge and compress your JavaScript and CSS files for faster website performance.**  Get started today and boost your web app's efficiency!

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features of Webassets:

*   **Asset Merging:** Combine multiple CSS and JavaScript files into single files, reducing HTTP requests.
*   **Asset Compression:** Minify your CSS and JavaScript files, decreasing file sizes and improving loading times.
*   **Streamlined Workflow:** Easy integration into your Python web development projects.
*   **Flexible Configuration:** Customize your asset management pipeline to fit your specific needs.

## Getting Started

### Documentation

Comprehensive documentation is available:

*   [Webassets Documentation](https://webassets.readthedocs.io/)

### Installation

It is recommended to use the latest code for webassets.

Install the development version using pip:

```bash
pip install webassets==dev
```

### Development

For development purposes, ensure you have the necessary tools and dependencies:

1.  **Prerequisites:** Java 7 or higher is required for certain features like the Google Closure filter.
2.  **Install Python Requirements:**
    ```bash
    uv venv
    uv pip install -r uv.lock
    ```
3.  **Install Other Requirements:**
    ```bash
    ./requirements-dev.sh
    ```
4.  **Run Tests:**
    ```bash
    ./run_tests.sh
    ```