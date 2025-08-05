# python-gitlab: Python Library for the GitLab API

**Effortlessly interact with GitLab through Python using the python-gitlab library.**

[View the original repository on GitHub](https://github.com/python-gitlab/python-gitlab)

This powerful Python package provides comprehensive access to the GitLab APIs, including:

**Key Features:**

*   **Pythonic Interface:** Manage GitLab resources with clean, easy-to-understand Python code.
*   **REST API Client:** A robust client for GitLab's v4 REST API.
*   **GraphQL API Support:** Synchronous and asynchronous clients for the GraphQL API.
*   **CLI Tool:** Includes a CLI tool (``gitlab``) that wraps REST API endpoints for convenient command-line access.
*   **Flexible Parameter Handling:** Pass any parameter supported by the GitLab API directly.
*   **Async Support:** Use the asynchronous client to prevent blocking.
*   **Lower-Level API Access:** Access new API endpoints as soon as they are available in GitLab.
*   **Persistent Sessions:** Utilize persistent request sessions for authentication, proxy, and certificate handling.
*   **Smart Error Handling:**  Includes smart retries for network and server errors, with built-in rate-limit handling.
*   **Paginated Response Handling:** Efficiently handle paginated responses with lazy iterators.
*   **Automatic Encoding:** Automatically URL-encodes paths and parameters.
*   **Data Conversion:** Automatically converts some complex data structures to API attribute types.
*   **Configuration:** Merge configuration from config files, environment variables, and arguments.

## Installation

``python-gitlab`` requires Python 3.9 or later.

Install the latest stable release using `pip`:

```bash
pip install --upgrade python-gitlab
```

To install from the latest development version, use:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

or from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using the Docker Images

``python-gitlab`` provides Docker images based on Alpine and Debian slim Python base images.

*   **Latest Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest`
*   **Latest Debian slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

Run the Docker image, for example:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Mount a config file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage Inside GitLab CI

Override the ``entrypoint`` in your GitLab CI configuration when using the Docker image:

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

### Building the Image

Build your own image from this repository:

```bash
docker build -t python-gitlab:latest .
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Gitter Community Chat

Get help from the community at [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby).

## Documentation

Comprehensive documentation is available at [http://python-gitlab.readthedocs.org/en/stable/](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for guidelines on contributing.