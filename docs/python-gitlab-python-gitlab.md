# python-gitlab: Python Client for GitLab APIs

**Effortlessly interact with the GitLab API using the python-gitlab library, empowering you to automate tasks and manage your GitLab resources with ease.**

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.org/project/python-gitlab/)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![GitHub license](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

`python-gitlab` is a powerful Python library designed for seamless interaction with the GitLab API. It provides a comprehensive client for both the v4 REST API and synchronous and asynchronous GraphQL APIs, along with a convenient CLI tool for interacting with REST API endpoints.

## Key Features

*   **Pythonic Interface:** Write clean and readable Python code to manage your GitLab resources.
*   **Flexible API Access:**  Pass arbitrary parameters to the GitLab API, accessing all available endpoints.
*   **GraphQL Support:** Utilize synchronous and asynchronous clients for the GraphQL API.
*   **Low-Level API Access:**  Access new endpoints immediately as they become available in GitLab.
*   **Persistent Sessions:** Leverage persistent request sessions for authentication, proxy handling, and certificate management.
*   **Robust Error Handling:** Benefit from smart retries for network and server errors, including rate-limit handling.
*   **Paginated Response Handling:** Efficiently manage paginated responses with lazy iterators.
*   **Automatic Encoding:** Automatically URL-encode paths and parameters as needed.
*   **Data Conversion:** Convert complex data structures to appropriate API attribute types automatically.
*   **Configuration Flexibility:** Merge configurations from configuration files, environment variables, and arguments.

## Installation

`python-gitlab` requires Python 3.9 or higher.

To install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

You can also install the development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

Or from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

Available tags include:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

Run the Docker image:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Mount your own configuration file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Using Inside GitLab CI:**

Override the `entrypoint` when using the image in GitLab CI:

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

**Building the Image:**

```bash
docker build -t python-gitlab:latest .
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at the [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues) page.

## Community

Join the community chat on [Gitter](https://gitter.im/python-gitlab/Lobby) to ask questions and discuss ideas.

## Documentation

The full documentation for CLI and API is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file for guidelines on contributing to `python-gitlab`.

**Find out more and contribute at the original repo: [https://github.com/python-gitlab/python-gitlab](https://github.com/python-gitlab/python-gitlab)**