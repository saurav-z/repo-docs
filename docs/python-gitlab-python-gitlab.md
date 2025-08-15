# python-gitlab: Python Library for Accessing GitLab APIs

**Easily interact with GitLab through Python with python-gitlab, providing both REST and GraphQL API access.**

[![Test](https://github.com/python-gitlab/python-gitlab/workflows/Test/badge.svg)](https://github.com/python-gitlab/python-gitlab/actions)
[![PyPI version](https://badge.fury.io/py/python-gitlab.svg)](https://badge.fury.io/py/python-gitlab)
[![Documentation Status](https://readthedocs.org/projects/python-gitlab/badge/?version=latest)](https://python-gitlab.readthedocs.org/en/latest/?badge=latest)
[![Codecov](https://codecov.io/github/python-gitlab/python-gitlab/coverage.svg?branch=main)](https://codecov.io/github/python-gitlab/python-gitlab?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-gitlab.svg)](https://pypi.python.org/pypi/python-gitlab)
[![Gitter](https://img.shields.io/gitter/room/python-gitlab/Lobby.svg)](https://gitter.im/python-gitlab/Lobby)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![GitHub license](https://img.shields.io/github/license/python-gitlab/python-gitlab)](https://github.com/python-gitlab/python-gitlab/blob/main/COPYING)

`python-gitlab` is a comprehensive Python package designed to provide robust access to GitLab's APIs. It empowers developers and DevOps engineers to automate and manage GitLab resources efficiently.

## Key Features

*   **Pythonic API:** Write clean and readable Python code to interact with GitLab.
*   **REST & GraphQL Support:** Utilize both REST (v4) and GraphQL APIs, offering flexibility in your approach.
*   **Asynchronous GraphQL Client:** Take advantage of asynchronous GraphQL support for improved performance.
*   **CLI Tool:** Use the included `gitlab` CLI tool to easily interact with REST API endpoints.
*   **Flexible Parameter Passing:** Pass any parameters supported by the GitLab API directly to the methods.
*   **Lower-Level API Access:** Access new endpoints as soon as they're available in GitLab.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Smart Retries & Rate Limiting:** Handle network errors and rate limits gracefully with built-in retry mechanisms.
*   **Pagination Handling:** Easily navigate paginated responses, including lazy iterators for efficient data retrieval.
*   **Automatic Encoding:** Automatically URL-encode paths and parameters to ensure correct API calls.
*   **Data Conversion:**  Automated conversion of certain complex data structures to API attribute types.
*   **Configuration Management:** Seamlessly merge configurations from config files, environment variables, and command-line arguments.

## Installation

`python-gitlab` requires Python 3.9 or higher.

Install the latest stable release using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.  The default tag is `alpine`.

Available tags:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

Run a Docker image:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

You can also mount your own configuration file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

When using the Docker image in GitLab CI, override the `entrypoint`:

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

### Building a Docker Image

Build your own image:

```bash
docker build -t python-gitlab:latest .
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests on the [GitHub issues page](https://github.com/python-gitlab/python-gitlab/issues).

## Community & Support

Join the [Gitter community](https://gitter.im/python-gitlab/Lobby) to ask questions and connect with other users.

## Documentation

Detailed documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

### Building the Documentation

Build the documentation using `tox`:

```bash
pip install tox
tox -e docs
```

## Contributing

Learn how to contribute to `python-gitlab` by referring to the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file.

**Visit the [python-gitlab GitHub repository](https://github.com/python-gitlab/python-gitlab) for the source code and more information.**