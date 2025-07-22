# Webassets: Powerful Asset Management for Python Web Development

**Webassets streamlines your web development workflow by effortlessly merging and compressing JavaScript and CSS files for optimal performance.** 

[View the original repository on GitHub](https://github.com/miracle2k/webassets)

## Key Features:

*   **Asset Bundling:** Efficiently combines multiple CSS and JavaScript files into single, optimized bundles.
*   **Compression:** Minifies and compresses your assets (CSS, JavaScript, and more) to reduce file sizes and improve load times.
*   **Flexible Integration:** Seamlessly integrates with various Python web frameworks.
*   **Extensible:** Supports a wide range of filters for advanced processing, such as LESS, Sass, CoffeeScript compilation, and image optimization.
*   **Production Ready:** Designed for real-world use in production environments, offering robust asset management capabilities.

## Installation

You can install Webassets using `pip`:

```bash
pip install webassets==dev
```

*Note: Releases are not on a regular schedule, so using the latest code is encouraged.*

## Documentation

Detailed documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

## Development

To contribute to Webassets, follow these steps:

1.  **Install Python requirements:**
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

    *Note: Running tests requires Java 7 (or later) for the Google Closure filter.*