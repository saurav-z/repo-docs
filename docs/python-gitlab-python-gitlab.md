# python-gitlab: Interact with GitLab APIs in Python

**Effortlessly manage your GitLab projects, resources, and workflows using the `python-gitlab` Python library.**

[View the original repository](https://github.com/python-gitlab/python-gitlab)

## Key Features

*   **Pythonic Interface:** Write clean, readable Python code to interact with GitLab.
*   **Comprehensive API Access:** Access GitLab's v4 REST and GraphQL APIs.
*   **GraphQL Support:** Use both synchronous and asynchronous GraphQL clients.
*   **CLI Tool:** Utilize the `gitlab` CLI for quick access to REST API endpoints.
*   **Flexible Parameter Handling:** Pass any parameters supported by the GitLab API.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy and certificate handling.
*   **Advanced Error Handling:** Leverage smart retries for network and server errors, including rate-limit handling.
*   **Pagination Support:** Easily navigate paginated responses with lazy iterators.
*   **Automatic Data Handling:** Enjoy automatic URL encoding and data structure conversion.
*   **Configuration Management:** Merge configuration from config files, environment variables, and command-line arguments.

## Installation

`python-gitlab` requires Python 3.9 or later.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest`
*   **Debian Slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

Run the image (e.g., to get a project):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

Mount a config file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage in GitLab CI

Override the `entrypoint` when using the Docker image in GitLab CI:

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

Report bugs and feature requests: [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues)

## Community

Join the [Gitter Community Chat](https://gitter.im/python-gitlab/Lobby) to ask questions and connect with other users.

## Documentation

Read the full documentation for the CLI and API: [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/)

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for guidelines on contributing to `python-gitlab`.