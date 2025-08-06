Here's an improved and SEO-optimized README for `webassets`, incorporating the requested features:

# Webassets: Streamline Your Web Development Asset Management

**Webassets simplifies and optimizes your web project by efficiently merging and compressing JavaScript and CSS files.**

[Link to Original Repo](https://github.com/miracle2k/webassets)

## Key Features

*   **Asset Bundling:**  Combine multiple CSS and JavaScript files into single, optimized bundles for faster loading times.
*   **Compression:** Minify CSS and JavaScript to reduce file sizes and improve performance.
*   **Flexible Integration:** Seamlessly integrate with various Python web frameworks.
*   **Extensible:** Easily add custom filters and processors.
*   **Well-Tested:** The project is well tested, see the CI badge above to ensure its stability.

## Installation

You can install `webassets` using pip:

```bash
pip install webassets==dev
```

or install the latest code via a tarball.

## Development

For development, you'll need to set up the development environment:

**Prerequisites:**

*   Python 3.x
*   Java 7 or later (required for some filters like Google Closure)

**Setup:**

1.  **Install Python requirements using uv:**

    ```bash
    uv venv
    uv pip install -r uv.lock
    ```

2.  **Install other requirements:**

    ```bash
    ./requirements-dev.sh
    ```

3.  **Run tests:**

    ```bash
    ./run_tests.sh