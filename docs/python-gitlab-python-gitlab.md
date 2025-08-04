# python-gitlab: Interact with GitLab APIs Effortlessly

**Python-gitlab is a powerful Python package enabling seamless interaction with GitLab APIs, offering both synchronous and asynchronous clients.**  Access the original repository [here](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic Code:** Write clean and readable Python code to manage your GitLab resources.
*   **Flexible API Access:** Pass any parameters directly to the GitLab API, leveraging the official GitLab documentation.
*   **Synchronous and Asynchronous GraphQL Support:** Choose the client type that best fits your needs.
*   **Direct Endpoint Access:** Access new GitLab API endpoints instantly via lower-level methods.
*   **Robust Session Management:** Utilize persistent request sessions for authentication, proxy, and certificate handling.
*   **Intelligent Error Handling:** Benefit from smart retries for network and server errors, including rate-limit handling.
*   **Paginated Response Handling:** Efficiently handle paginated responses with lazy iterators for optimal performance.
*   **Automatic Data Handling:** Automatically URL-encode paths and parameters and convert complex data structures.
*   **Configuration Flexibility:** Merge configurations from config files, environment variables, and command-line arguments.

## Installation

Ensure you have Python 3.9 or higher installed.  Use pip to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

You can also install the development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

**Available Tags:**

*   `latest` (Alpine)
*   `alpine` (latest Alpine)
*   `slim-bullseye` (latest Debian slim)
*   `v3.2.0` (Alpine)
*   `v3.2.0-alpine`
*   `v3.2.0-slim-bullseye`

**Running a Docker Container:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Using in GitLab CI:**  (Requires overriding the entrypoint, see original README for details)

## Bug Reports and Support

Report bugs and feature requests on [GitHub](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the Gitter community for quick questions and discussions: [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby)

## Documentation

Comprehensive documentation for both the CLI and API is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file for contribution guidelines.