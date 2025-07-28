# python-gitlab: Interact with the GitLab API in Python

**python-gitlab** is a powerful Python library that simplifies interaction with GitLab's API, allowing you to manage your GitLab resources with ease.  For more information, please visit the original repository: [https://github.com/python-gitlab/python-gitlab](https://github.com/python-gitlab/python-gitlab)

## Key Features

*   **Pythonic API:** Write clean, readable Python code to interact with GitLab.
*   **Comprehensive API Coverage:** Access both the v4 REST API and synchronous/asynchronous GraphQL APIs.
*   **Flexible Parameter Handling:** Pass arbitrary parameters to the GitLab API, leveraging GitLab's documentation.
*   **Asynchronous GraphQL Support:** Utilize asynchronous clients for enhanced performance.
*   **Low-Level API Access:** Access new GitLab endpoints as soon as they are available.
*   **Persistent Sessions:** Leverage persistent requests sessions for authentication, proxy, and certificate handling.
*   **Smart Retries and Rate Limiting:** Handle network and server errors gracefully, including rate-limit handling.
*   **Paginated Response Handling:** Work with paginated responses using lazy iterators.
*   **Automatic Encoding:** Automatically URL-encode paths and parameters.
*   **Data Conversion:** Automatically convert some complex data structures to API attribute types.
*   **Configuration Management:** Merge configuration from config files, environment variables, and command-line arguments.
*   **CLI Tool:** A CLI tool is provided to wrap the REST API endpoints.

## Installation

``python-gitlab`` is compatible with Python 3.9+.

To install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or install the latest development version directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

``python-gitlab`` provides Docker images for easy use, based on the Alpine and Debian slim python base images. You can choose the image that best fits your needs.

The images are published on the GitLab registry.

### Example Usage

*   Run a specific command:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

*   Mount your own config file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

If you're using the Docker image in your GitLab CI, override the entrypoint:

```yaml
Job Name:
   image:
      name: registry.gitlab.com/python-gitlab/python-gitlab:latest
      entrypoint: [""]
   before_script:
      gitlab --version
   script:
      gitlab <command>
```

## Bug Reports & Community

*   **Report Bugs:** Report issues and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).
*   **Community Chat:** Join the Gitter community chat for discussions and support: [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby).

## Documentation

The full documentation is available on Read the Docs: [http://python-gitlab.readthedocs.org/en/stable/](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to `CONTRIBUTING.rst <https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst>`__ for guidelines on contributing.