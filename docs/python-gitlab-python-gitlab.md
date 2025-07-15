# python-gitlab: Interact with GitLab APIs Effortlessly

**python-gitlab** is a powerful Python package that simplifies interacting with GitLab's API, allowing you to manage your GitLab resources with ease.  Check out the original repository [here](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic Code:** Write clean, readable Python code to manage your GitLab projects, users, and more.
*   **Flexible API Access:** Pass arbitrary parameters to the GitLab API, adhering to GitLab's documentation.
*   **Synchronous and Asynchronous GraphQL Support:** Utilize both synchronous and asynchronous clients for GraphQL API interactions.
*   **Comprehensive API Coverage:** Access any GitLab endpoint, even newly released ones, using lower-level API methods.
*   **Robust Session Management:** Benefit from persistent request sessions for authentication, proxy settings, and certificate handling.
*   **Smart Error Handling:** Experience intelligent retries for network and server errors, including rate-limit management.
*   **Efficient Pagination:** Seamlessly handle paginated responses with lazy iterators for improved performance.
*   **Automatic Data Handling:** Enjoy automatic URL encoding and data type conversions for complex data structures.
*   **Configuration Flexibility:** Merge settings from configuration files, environment variables, and command-line arguments.

## Installation

**Prerequisites:** Python 3.9+

Use `pip` to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Or, install the latest development version directly from GitHub or GitLab:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest`
*   **Debian Slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

**Run a Docker Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example:** Get a project from GitLab.com (without authentication):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mount a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage Inside GitLab CI

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

## Build the Docker image

To build your own image from this repository, run:

```bash
docker build -t python-gitlab:latest .
```

Run your own image:

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at: [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues)

## Community

Join the [Gitter Community Chat](https://gitter.im/python-gitlab/Lobby) for quick questions and discussions.

## Documentation

*   Full documentation: [http://python-gitlab.readthedocs.org/en/stable/](http://python-gitlab.readthedocs.org/en/stable/)

## Contributing

Refer to `CONTRIBUTING.rst <https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst>`_ for contribution guidelines.