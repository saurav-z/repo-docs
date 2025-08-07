# webassets: Streamline Web Asset Management for Python

**webassets empowers Python web developers to effortlessly merge and compress JavaScript and CSS files, optimizing website performance and streamlining development workflows.** For more in-depth information, check out the [original repository](https://github.com/miracle2k/webassets).

## Key Features

*   **Asset Bundling:** Merge multiple JavaScript and CSS files into single, optimized bundles.
*   **Compression:** Automatically compress your assets to reduce file sizes and improve loading times.
*   **Extensible:** Supports a variety of filters for pre-processing, minification, and more.
*   **Production-Ready:** Designed for use in production environments to ensure optimal website performance.

## Installation

Install the development version using pip:

```bash
pip install webassets==dev
```

## Development

To contribute to webassets, you'll need to set up your development environment.

**Prerequisites:**

*   Python 3.7+
*   Java 7+ (required for Google Closure filter)

**Steps:**

1.  **Create a virtual environment using uv:**

    ```bash
    uv venv
    ```

2.  **Install Python dependencies using uv:**

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

## Documentation

Comprehensive documentation is available at: [https://webassets.readthedocs.io/](https://webassets.readthedocs.io/)

---

**Note:**  While releases may not be on a strict schedule, the project is well-tested.  The build status badge at the top of the [original repository](https://github.com/miracle2k/webassets) provides a quick indicator of build health.