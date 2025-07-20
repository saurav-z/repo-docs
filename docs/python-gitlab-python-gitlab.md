# python-gitlab: Interact with GitLab APIs in Python

**Python-gitlab provides a powerful and flexible way to interact with GitLab APIs, enabling you to automate your GitLab workflows.** [Explore the python-gitlab project on GitHub](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic GitLab API access:** Write clean, readable Python code to manage GitLab resources.
*   **Full API coverage:** Access all GitLab API endpoints, including v4 REST and GraphQL, both synchronously and asynchronously.
*   **Arbitrary parameter support:** Pass any parameters supported by the GitLab API directly.
*   **Asynchronous GraphQL client:** Utilize an async client when working with the GraphQL API.
*   **Lower-level API methods:** Access endpoints as soon as they become available in GitLab.
*   **Persistent sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust error handling:** Automatic retries for network and server errors, with rate-limit handling.
*   **Efficient pagination:** Flexible handling of paginated responses, including lazy iterators.
*   **Automatic data handling:** Automatically URL-encodes paths and parameters and converts complex data structures to API attribute types.
*   **Configuration management:** Merge configurations from config files, environment variables, and command-line arguments.

## Installation

python-gitlab requires Python 3.9 or later.

Use `pip` to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

python-gitlab offers Docker images in both Alpine and Debian slim flavors. The Alpine image is the default. The Debian-based slim tag is a good alternative if you need a more complete environment with a bash shell.

**Available Images:**

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

**Example (get a project from GitLab.com):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

To use the Docker image directly inside your GitLab CI jobs, override the `entrypoint`:

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

To build your own image:

```bash
docker build -t python-gitlab:latest .
```

**Building a Debian slim-based image:**

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Please report bugs and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Gitter Community Chat

Get help from the community and discuss the project on [Gitter](https://gitter.im/python-gitlab/Lobby).

## Documentation

Full documentation is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for guidelines.