# python-gitlab: Interact with the GitLab API in Python

**python-gitlab** is a powerful Python package that makes it easy to interact with the GitLab API, offering comprehensive features for managing your GitLab resources. ([View on GitHub](https://github.com/python-gitlab/python-gitlab))

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

## Key Features

*   **Pythonic Interface:** Write clean, readable Python code to manage your GitLab resources.
*   **Flexible API Access:** Pass any parameters to the GitLab API, following GitLab's documentation.
*   **GraphQL Support:** Use synchronous or asynchronous clients for the GraphQL API.
*   **Low-Level API Access:** Access new API endpoints quickly using lower-level methods.
*   **Persistent Sessions:** Utilize persistent request sessions for authentication and proxy handling.
*   **Robust Error Handling:** Benefit from smart retries on network and server errors, including rate-limit handling.
*   **Pagination Handling:** Easily handle paginated responses with lazy iterators.
*   **Automatic Encoding:** URL-encode paths and parameters automatically.
*   **Data Conversion:** Automatically convert complex data structures to API attribute types.
*   **Configuration Management:** Merge configuration from config files, environment variables, and arguments.
*   **CLI Tool:** Includes a command-line interface (``gitlab``) for interacting with the REST API.

## Installation

**Prerequisites:**  Python 3.9+

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Install the development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** Smaller image.
*   **Debian Slim:** (e.g., `-slim-bullseye`)  For environments needing a more complete setup, e.g., bash shell in CI jobs.

**Available Images:**

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (Alpine, alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (Debian Slim)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (Alpine, alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

**Running the Docker Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example (without authentication):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Config File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Usage in GitLab CI:**
Override the `entrypoint` in your GitLab CI job definition, as per the official documentation:

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
**Building Your Own Image:**

```bash
docker build -t python-gitlab:latest .
docker run -it --rm python-gitlab:latest <command> ...
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests on [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community

Join the [Gitter Community Chat](https://gitter.im/python-gitlab/Lobby) for discussions and support.

## Documentation

Comprehensive documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.