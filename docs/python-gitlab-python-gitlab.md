# python-gitlab: Effortlessly Interact with the GitLab API

**python-gitlab** is a powerful Python package designed to simplify your interactions with the GitLab API, providing both synchronous and asynchronous clients, and a helpful CLI tool.  Access the original repository for more information [here](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic GitLab Management:** Write clean, readable Python code to manage GitLab resources.
*   **Flexible API Interaction:** Pass arbitrary parameters to the GitLab API, mirroring GitLab's own documentation.
*   **Asynchronous and Synchronous GraphQL Support:** Utilize either type of client for the GraphQL API based on your needs.
*   **Access to Latest Endpoints:**  Access new GitLab API endpoints as soon as they're available, using lower-level API methods.
*   **Persistent Sessions & Error Handling:** Benefit from persistent request sessions for authentication, proxy, and certificate handling, alongside smart retries and rate-limit handling.
*   **Paginated Response Handling:** Efficiently handle paginated responses with lazy iterators.
*   **Automatic Data Handling:** Automatically URL-encode paths and parameters and convert complex data structures to API attribute types.
*   **Configuration Flexibility:**  Merge configurations seamlessly from config files, environment variables, and command-line arguments.
*   **CLI Tool:** Includes a helpful command-line interface for interacting with the GitLab REST API.

## Installation

`python-gitlab` requires Python 3.9 or later.

To install using `pip`:

```bash
pip install --upgrade python-gitlab
```

You can also install directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

Or, from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

**Available Tags:**

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

**Run a Docker Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example:** Get a project from GitLab.com (without authentication):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mount a Config File:**

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

### Building Your Own Image

Build from the repository:

```bash
docker build -t python-gitlab:latest .
```

Run your custom image:

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Reporting Issues

Report bugs and feature requests on GitHub:  [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues)

## Community & Documentation

*   **Gitter Community Chat:** [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby) - Get help with questions.
*   **Documentation:**  [http://python-gitlab.readthedocs.org/en/stable/](http://python-gitlab.readthedocs.org/en/stable/)

## Contributing

Refer to the `CONTRIBUTING.rst` file on GitHub for guidelines.