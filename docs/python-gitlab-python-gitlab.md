# Python-GitLab: Interact with GitLab APIs Easily

**Python-GitLab** is a powerful Python library that simplifies interaction with GitLab's APIs, enabling you to manage your GitLab resources programmatically.  For more in-depth information, visit the original repository: [https://github.com/python-gitlab/python-gitlab](https://github.com/python-gitlab/python-gitlab)

## Key Features

*   **Pythonic Interface:** Write clean, readable Python code to interact with GitLab.
*   **Full API Coverage:** Access all GitLab API endpoints, including v4 REST and GraphQL, both synchronously and asynchronously.
*   **Flexible Parameter Handling:** Pass arbitrary parameters to the GitLab API, utilizing the official GitLab documentation for guidance.
*   **Asynchronous Support:** Choose between synchronous and asynchronous clients for GraphQL API access.
*   **Low-Level API Access:** Access new endpoints as soon as they are available on GitLab.
*   **Persistent Sessions:** Leverage persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust Error Handling:** Benefit from smart retries on network and server errors, along with rate-limit handling.
*   **Paginated Response Handling:**  Manage paginated responses with ease, including lazy iterators.
*   **Automatic Encoding & Conversion:** Paths and parameters are automatically URL-encoded, and complex data structures are converted to API-compatible types.
*   **Configuration Flexibility:** Merge configurations from config files, environment variables, and command-line arguments.

## Installation

Python-GitLab requires Python 3.9 or later.

Install the latest stable version using pip:

```bash
pip install --upgrade python-gitlab
```

Or, install the development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

Python-GitLab offers Docker images in two flavors: Alpine (default) and Debian slim. Use the slim image if you require a more complete environment with a bash shell.

Available images include:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest Debian slim)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

Run a Docker image (e.g., to get a project):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

Mount a configuration file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

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

## Bug Reports and Feature Requests

Report issues and suggestions at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the Gitter community at [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby) for discussions and support.

## Documentation

Find comprehensive documentation for both the CLI and API on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

### Building the Documentation

Build the documentation locally using tox:

```bash
pip install tox
tox -e docs
```

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.