# Python-GitLab: Interact with GitLab APIs Effortlessly

**Python-GitLab** is a powerful Python library that simplifies interaction with GitLab's API, allowing you to manage your GitLab resources with ease.  Access the original repository [here](https://github.com/python-gitlab/python-gitlab).

[![Build Status](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Code Coverage](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

## Key Features

*   **Pythonic API:** Write clean, readable Python code to manage your GitLab resources.
*   **Comprehensive API Coverage:** Access all GitLab API endpoints, including v4 REST and GraphQL.
*   **Asynchronous GraphQL Support:** Utilize asynchronous clients for non-blocking GraphQL interactions.
*   **Flexible Parameter Handling:** Pass any parameters supported by the GitLab API directly.
*   **Persistent Sessions:** Maintain persistent sessions for authentication, proxy, and certificate management.
*   **Smart Error Handling:** Benefit from automatic retries on network and server errors, including rate-limit handling.
*   **Pagination Handling:** Efficiently handle paginated responses with lazy iterators.
*   **Automatic Encoding:**  Automatic URL encoding for paths and parameters.
*   **Configuration Management:** Merge configuration from files, environment variables, and arguments.

## Installation

Python-GitLab requires Python 3.9 or higher.

Install using pip:

```bash
pip install --upgrade python-gitlab
```

Install the development version from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

Python-GitLab provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest` (aliased as `alpine`)
*   **Debian slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

Run the image:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Mount a config file:

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

## Bug Reports

Report bugs and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community

Join the [Gitter community chat](https://gitter.im/python-gitlab/Lobby) for discussions and help.

## Documentation

Comprehensive documentation is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contributing guidelines.