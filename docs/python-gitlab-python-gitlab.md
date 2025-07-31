# python-gitlab: Interact with GitLab APIs in Python

**python-gitlab** is a powerful Python library that simplifies interaction with GitLab's APIs, enabling developers to automate tasks and manage GitLab resources efficiently. ([Original Repository](https://github.com/python-gitlab/python-gitlab))

[![Tests](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Code Coverage](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![GitHub license](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

## Key Features

*   **Pythonic API:** Write clean, readable Python code to interact with GitLab.
*   **Comprehensive API Coverage:** Access both REST and GraphQL APIs (synchronous and asynchronous).
*   **Arbitrary Parameter Passing:**  Pass any parameter supported by the GitLab API directly.
*   **Flexible Client Options:** Choose between synchronous and asynchronous clients for GraphQL.
*   **Low-Level Access:** Access new API endpoints quickly using low-level methods.
*   **Persistent Sessions:** Utilize persistent request sessions for authentication and proxy support.
*   **Robust Error Handling:** Includes smart retries and rate-limit handling.
*   **Pagination Handling:** Easily navigate paginated responses with lazy iterators.
*   **Automatic Encoding:** Automatically URL-encodes paths and parameters.
*   **Data Conversion:** Automatically converts some complex data structures to API attribute types.
*   **Configuration Management:**  Merges configuration from config files, environment variables, and arguments.
*   **CLI Tool:** A command-line interface (``gitlab``) to interact with the REST API.

## Installation

**Prerequisites:** Python 3.9 or higher.

Use `pip` to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install the development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest) or `registry.gitlab.com/python-gitlab/python-gitlab:alpine`
*   **Debian Slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

Run a container:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Mount a configuration file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage in GitLab CI

To use the Docker image in GitLab CI, override the `entrypoint`:

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

Build your own image:

```bash
docker build -t python-gitlab:latest .
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at: [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues)

## Community

Join the [Gitter Community Chat](https://gitter.im/python-gitlab/Lobby) for discussions and support.

## Documentation

Full documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.