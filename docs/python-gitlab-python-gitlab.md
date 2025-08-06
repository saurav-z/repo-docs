<!-- SEO-optimized README for python-gitlab -->

# python-gitlab: Interact with GitLab APIs in Python

**Effortlessly manage your GitLab resources with the powerful and versatile python-gitlab library.**

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Code Coverage](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![GitHub license](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

`python-gitlab` is a comprehensive Python package designed to provide robust access to the GitLab APIs.  It offers a versatile toolkit for interacting with GitLab, suitable for automating tasks, developing integrations, and building custom solutions.

## Key Features

*   **Pythonic Interface:** Write clean, readable Python code for managing GitLab resources.
*   **REST and GraphQL API Clients:** Access GitLab's v4 REST API, and both synchronous and asynchronous GraphQL API clients.
*   **Flexible Parameter Passing:**  Pass arbitrary parameters to the GitLab API, using GitLab's own documentation as a guide.
*   **Asynchronous GraphQL Support:** Leverage asynchronous clients for efficient GraphQL API interactions.
*   **Lower-Level API Access:** Access new endpoints and features as soon as they are available in GitLab via lower-level API methods.
*   **Persistent Session Handling:** Utilize persistent request sessions for secure authentication, proxy support, and certificate management.
*   **Intelligent Error Handling:** Benefit from smart retries on network and server errors, including rate-limit management.
*   **Pagination and Iteration:** Efficiently handle paginated responses with lazy iterators.
*   **Automatic Encoding and Conversion:** Enjoy automated URL encoding and data type conversions for ease of use.
*   **Configuration Flexibility:** Easily merge configurations from config files, environment variables, and command-line arguments.

## Installation

`python-gitlab` requires Python 3.9 or higher. Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

Or from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images. You can find them on the GitLab registry.

```bash
# Example: Run a command against a public project
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

## Usage in GitLab CI

Use the Docker image within your GitLab CI jobs by overriding the `entrypoint`:

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

## Building the Image

Build your own Docker image:

```bash
docker build -t python-gitlab:latest .
```

Run your custom image:

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

## Reporting Issues

Report bugs and suggest features at [the GitHub issues page](https://github.com/python-gitlab/python-gitlab/issues).

## Community & Support

Join the `Gitter community chat <https://gitter.im/python-gitlab/Lobby>`_ for discussions and support.

## Documentation

Comprehensive documentation is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file for contribution guidelines.

**[Visit the python-gitlab repository on GitHub](https://github.com/python-gitlab/python-gitlab) for more details.**