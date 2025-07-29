# python-gitlab: Python Library for the GitLab API

**Easily interact with GitLab through a powerful and flexible Python library!**  Visit the original repository [here](https://github.com/python-gitlab/python-gitlab).

This library provides comprehensive access to GitLab's APIs, enabling you to manage your projects, users, and other GitLab resources programmatically. It includes client implementations for:

*   **REST API v4:** Access the core GitLab functionalities through a synchronous client.
*   **GraphQL API:** Utilize synchronous and asynchronous clients for enhanced query capabilities.
*   **Command-Line Interface (CLI):**  A convenient `gitlab` CLI tool to wrap REST API endpoints.

## Key Features

*   **Pythonic Interface:** Write clean and readable Python code to interact with GitLab.
*   **Flexible API Interaction:** Pass custom parameters to the GitLab API, utilizing all available options documented by GitLab.
*   **GraphQL Support:** Leverage both synchronous and asynchronous clients for GraphQL operations.
*   **Full API Coverage:** Access new GitLab endpoints as soon as they are available using lower-level API methods.
*   **Persistent Sessions:** Benefit from persistent request sessions for secure authentication and efficient proxy/certificate handling.
*   **Robust Error Handling:** Includes smart retries for network and server errors, along with rate-limit handling.
*   **Pagination Management:** Effortlessly handle paginated responses with lazy iterators.
*   **Automated Data Handling:** Automatically encode URLs and convert complex data structures for seamless API interactions.
*   **Configuration Flexibility:** Merge settings from configuration files, environment variables, and command-line arguments.

## Installation

Requires Python 3.9 or higher. Use pip to install the latest stable release:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install directly from the GitHub or GitLab repositories for the development version:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git  # From GitHub
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git # From GitLab
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images. Use these images for easy integration in CI/CD pipelines or for containerized deployments.

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

**Example:** Get a project from GitLab.com (without authentication):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Using in GitLab CI:**
Remember to override the `entrypoint` in your `.gitlab-ci.yml` file.

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

**Building Your Own Image:**

```bash
docker build -t python-gitlab:latest .
docker run -it --rm python-gitlab:latest <command> ...
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye . # Build Debian slim image
```

## Reporting Issues and Getting Help

*   **Report bugs and request features:**  [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues)
*   **Join the community:**  [Gitter Chat](https://gitter.im/python-gitlab/Lobby)

## Documentation

Comprehensive documentation is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See the [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) file for contribution guidelines.