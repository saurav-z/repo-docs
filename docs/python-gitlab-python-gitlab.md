# python-gitlab: A Powerful Python Library for Interacting with the GitLab API

**Effortlessly automate and manage your GitLab resources with the python-gitlab library.**  Learn more about [python-gitlab on GitHub](https://github.com/python-gitlab/python-gitlab).

## Key Features:

*   **Pythonic GitLab API Access:** Interact with GitLab using intuitive, Pythonic code.
*   **Flexible API Parameter Passing:**  Pass any parameters directly to the GitLab API, following GitLab's official documentation.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for the GraphQL API.
*   **Comprehensive API Coverage:** Access any GitLab endpoint as soon as it is available, using lower-level API methods.
*   **Robust Session Management:** Benefit from persistent request sessions for authentication, proxy settings, and certificate handling.
*   **Intelligent Error Handling:**  Includes smart retries for network and server errors, along with rate-limit handling.
*   **Efficient Pagination:**  Handles paginated responses effectively, including lazy iterators.
*   **Automatic Data Encoding:** Automatically URL-encodes paths and parameters.
*   **Data Structure Conversion:** Automatically converts complex data structures to appropriate API attribute types.
*   **Flexible Configuration:**  Merge configuration from config files, environment variables, and command-line arguments.

## Installation

``python-gitlab`` requires Python 3.9 or later.

Use `pip` to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Or, install the latest development version directly from the Git repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images built on Alpine and Debian slim Python base images.  The default tag is `alpine`.

Available tags:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

To run the Docker image:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Example:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

Mount a configuration file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Using in GitLab CI

Override the `entrypoint` to use the Docker image in GitLab CI.

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

To build your own image from the repository:

```bash
docker build -t python-gitlab:latest .
```

Run the image:

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at:  https://github.com/python-gitlab/python-gitlab/issues

## Community Chat

Join the community on Gitter: https://gitter.im/python-gitlab/Lobby

## Documentation

Complete documentation is available on Read the Docs:  http://python-gitlab.readthedocs.org/en/stable/

## Contributing

Refer to `CONTRIBUTING.rst <https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst>`_ for contribution guidelines.