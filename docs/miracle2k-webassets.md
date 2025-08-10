# webassets: Streamline Your Python Web Development with Powerful Asset Management

**webassets** simplifies your Python web development workflow by efficiently merging and compressing your JavaScript and CSS files, resulting in faster loading times and improved website performance.  [Check out the original repository](https://github.com/miracle2k/webassets).

## Key Features:

*   **Asset Bundling:** Merge multiple JavaScript and CSS files into single, optimized bundles.
*   **Compression:**  Minimize file sizes with built-in compression for CSS and JavaScript, improving loading speeds.
*   **Integration:** Seamlessly integrates with various Python web frameworks.
*   **Flexible:** Offers extensive customization options to tailor asset management to your specific needs.
*   **Active Development:** Benefit from ongoing development and improvements.

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Installation

While releases are not on a strict schedule, the codebase is well-tested. You can install the latest development version using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to webassets or run the test suite, follow these steps:

1.  **Install Python Requirements:**
    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install Other Requirements:**
    ```bash
    ./requirements-dev.sh
    ```

3.  **Run Tests:**
    ```bash
    ./run_tests.sh
    ```

**Note:** Running tests requires Java 7 (or later) installed, as it's a dependency for the Google Closure filter.

## Contributing

We welcome contributions!  Feel free to submit pull requests or open issues on the [GitHub repository](https://github.com/miracle2k/webassets).