# python-gitlab: Your Pythonic Gateway to GitLab APIs

**Effortlessly manage your GitLab projects and resources with `python-gitlab`, a robust Python library providing access to GitLab's powerful APIs.**  [Explore the original repository](https://github.com/python-gitlab/python-gitlab) for detailed information and contributions.

## Key Features

*   **Pythonic API Access:** Interact with GitLab using intuitive Python code.
*   **Comprehensive API Coverage:** Supports GitLab's v4 REST API, and synchronous and asynchronous GraphQL APIs.
*   **Flexible API Interaction:** Pass custom parameters to any GitLab API endpoint.
*   **Asynchronous and Synchronous GraphQL Support:** Choose the right client for your needs.
*   **Lower-Level Access:** Access new endpoints as soon as they are available on GitLab.
*   **Persistent Sessions:** Leverage persistent request sessions for efficient authentication, proxy, and certificate handling.
*   **Smart Error Handling:** Benefit from automatic retries, rate-limit handling, and more.
*   **Advanced Pagination:** Navigate paginated responses with ease, including lazy iterators.
*   **Automatic Data Conversion:**  Complex data structures converted to API attribute types
*   **Configuration Flexibility:** Merge configuration from config files, environment variables, and command-line arguments.
*   **CLI Tool:** Utilize a command-line interface (``gitlab``) for quick access to REST API endpoints.

## Installation

`python-gitlab` requires Python 3.9 or higher.

Use pip to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Or, install the latest development version directly from the source:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images, available on the GitLab registry. The default tag is ``alpine``.

**Available Docker Image Tags:**

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

**Example:** Get a project (without authentication):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage Inside GitLab CI

To use the Docker image within GitLab CI, you may need to override the image's entrypoint:

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

Build your own Docker image:

```bash
docker build -t python-gitlab:latest .
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Reporting Bugs and Feature Requests

Submit bugs and feature requests on the project's [issue tracker](https://github.com/python-gitlab/python-gitlab/issues).

## Community

Join the vibrant `python-gitlab` community on [Gitter](https://gitter.im/python-gitlab/Lobby) to ask questions, share ideas, and connect with other users.

## Documentation

Find comprehensive documentation for both the CLI and API on [ReadTheDocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Review the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file for guidelines on contributing to `python-gitlab`.