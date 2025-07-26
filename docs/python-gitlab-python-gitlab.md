# Python-GitLab: Interact with GitLab APIs with Ease

**Python-GitLab** is a powerful Python library that simplifies interacting with GitLab APIs, providing both synchronous and asynchronous options. ([See the original repository](https://github.com/python-gitlab/python-gitlab))

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Codecov](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

## Key Features

Python-GitLab offers a range of features to make working with GitLab APIs straightforward:

*   **Pythonic Interface:** Write clean, readable Python code to manage your GitLab resources.
*   **Flexible API Access:** Pass arbitrary parameters to the GitLab API, as documented by GitLab.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for the GraphQL API.
*   **Low-Level API Access:** Access new GitLab API endpoints quickly through lower-level methods.
*   **Persistent Sessions:** Use persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust Error Handling:** Benefit from smart retries on network and server errors, along with rate-limit handling.
*   **Paginated Response Handling:** Easily manage paginated responses, including lazy iterators.
*   **Automatic Encoding:** Paths and parameters are automatically URL-encoded.
*   **Data Type Conversion:** Automatically converts complex data structures to API attribute types.
*   **Configuration Management:** Merge configuration from config files, environment variables, and arguments.

## Installation

Python-GitLab requires Python 3.9 or later.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

Python-GitLab provides Docker images based on Alpine and Debian slim Python base images.  This allows for quick and easy access to the tools.

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

## Usage inside GitLab CI

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

## Bug Reports

Report bugs and feature requests on [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community

Join the [Gitter community](https://gitter.im/python-gitlab/Lobby) for help and discussions.

## Documentation

Comprehensive documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contributing guidelines.