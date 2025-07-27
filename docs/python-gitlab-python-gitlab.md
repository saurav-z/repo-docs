<!-- python-gitlab README.md -->

# python-gitlab: Python Library for GitLab API Access

**Effortlessly interact with the GitLab API using the powerful and versatile `python-gitlab` library.**  Access the original repository on [GitHub](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic GitLab Management:** Write clean and readable Python code to manage GitLab resources.
*   **Flexible API Parameter Passing:** Pass any parameter supported by the GitLab API.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for GitLab's GraphQL API.
*   **Access to All Endpoints:** Access GitLab endpoints as soon as they're available.
*   **Persistent Session Handling:** Leverage persistent request sessions for authentication, proxy, and certificate management.
*   **Smart Retry Mechanisms:** Benefit from smart retries for network and server errors with rate-limit handling.
*   **Pagination Handling:** Easily handle paginated responses with lazy iterators.
*   **Automatic URL Encoding:** Automatically encode paths and parameters where necessary.
*   **Data Structure Conversion:** Automatically converts complex data structures to API attribute types.
*   **Configuration Flexibility:** Merge configurations from config files, environment variables, and command-line arguments.

## Installation

`python-gitlab` is compatible with Python 3.9 and later.

Install the latest stable version using pip:

```bash
pip install --upgrade python-gitlab
```

Or, install directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (Latest Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (Latest Debian slim)

**Run the Docker image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example (without authentication):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mount a configuration file:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage in GitLab CI

Override the entrypoint when using the Docker image in GitLab CI:

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

### Building Your Own Image

```bash
docker build -t python-gitlab:latest .  # Build Alpine-based image
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye . # Build Debian slim-based image
```

## Get Help

*   **Bug Reports & Feature Requests:** [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues)
*   **Community Chat:** [Gitter](https://gitter.im/python-gitlab/Lobby)
*   **Documentation:** [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/)

## Contributing

Refer to `CONTRIBUTING.rst <https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst>` for contribution guidelines.