# python-gitlab: Interact with GitLab APIs using Python

**python-gitlab** is the go-to Python package for seamlessly interacting with GitLab's REST and GraphQL APIs.  For the original source code, visit the [python-gitlab GitHub repository](https://github.com/python-gitlab/python-gitlab).

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Codecov](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![GitHub license](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

## Key Features

*   **Pythonic API:**  Write clean and readable Python code to manage your GitLab resources.
*   **Flexible Parameter Handling:** Pass any parameters directly to the GitLab API, mirroring the official documentation.
*   **GraphQL API Support:** Use synchronous or asynchronous clients for accessing the GraphQL API.
*   **Low-Level API Access:** Access new GitLab API endpoints quickly via lower-level methods.
*   **Persistent Sessions:**  Utilize persistent request sessions for authentication, proxy settings, and certificate handling.
*   **Robust Error Handling:** Includes smart retries for network and server errors, with integrated rate-limit handling.
*   **Paginated Response Handling:** Easily navigate through paginated responses with lazy iterators.
*   **Automatic Encoding:**  Handles URL encoding of paths and parameters automatically.
*   **Data Type Conversion:** Automatically converts complex data structures to API-compatible attribute types.
*   **Configuration Management:** Merge configuration from configuration files, environment variables, and command-line arguments.

## Installation

**Requirements:** Python 3.9 or later

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or install directly from the Git repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

**Available Tags:**

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

**Running the Docker Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example (Get a project without authentication):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Using in GitLab CI:**
When using the Docker image directly in GitLab CI, you will need to override the entrypoint, as noted in the official GitLab documentation.

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
To build your own image from the repository, run:
```bash
docker build -t python-gitlab:latest .
```

**Build a Debian slim-based image:**
```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests on the [GitHub issues page](https://github.com/python-gitlab/python-gitlab/issues).

## Community

Join the python-gitlab community chat on [Gitter](https://gitter.im/python-gitlab/Lobby) for support and discussions.

## Documentation

Comprehensive documentation for both the CLI and API is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Learn how to contribute by reading the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file.