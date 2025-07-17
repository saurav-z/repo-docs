# python-gitlab: Interact with GitLab APIs in Python

**python-gitlab** is a powerful Python package that simplifies interacting with GitLab APIs, enabling you to automate tasks, manage resources, and build custom integrations.  [Explore the original repository](https://github.com/python-gitlab/python-gitlab).

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Code Coverage](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![GitHub license](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

## Key Features

*   **Comprehensive API Access:** Access GitLab's v4 REST API and both synchronous and asynchronous GraphQL APIs.
*   **Pythonic Code:** Write clean, readable Python code to manage GitLab resources.
*   **Flexible Parameter Handling:** Pass any valid parameters directly to the GitLab API, according to the official GitLab documentation.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for GraphQL API interactions.
*   **Lower-Level API Access:** Access new GitLab API endpoints as soon as they are available using lower-level API methods.
*   **Persistent Sessions:** Leverage persistent request sessions for secure authentication, proxy settings, and certificate handling.
*   **Robust Error Handling:** Benefit from smart retry mechanisms for network and server errors, including rate-limit handling.
*   **Paginated Response Handling:** Easily navigate paginated responses with flexible handling, including lazy iterators.
*   **Automatic Encoding:** Paths and parameters are automatically URL-encoded as needed.
*   **Data Conversion:** Complex data structures are automatically converted to API attribute types.
*   **Configuration Flexibility:** Merge configuration from config files, environment variables, and command-line arguments.
*   **CLI tool**: Manage your GitLab via command line.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

You can also install directly from the GitHub or GitLab repositories:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

or:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Docker Images

**python-gitlab** provides Docker images based on Alpine and Debian slim Python base images for convenient use.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest`
*   **Debian Slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (for more complete environment or running into issues)

Run a container:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

## Usage in GitLab CI

To use the Docker image in GitLab CI, override the `entrypoint`:

```yaml
Job Name:
   image:
      name: registry.gitlab.com/python-gitlab/python-gitlab:latest
      entrypoint: [""]
   before_script:
      - gitlab --version
   script:
      - gitlab <command>
```

## Building Docker Images

Build your own image:

```bash
docker build -t python-gitlab:latest .
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Reporting Issues

Report bugs and feature requests on [GitHub](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the [Gitter community](https://gitter.im/python-gitlab/Lobby) for discussions and support.

## Documentation

Full documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.