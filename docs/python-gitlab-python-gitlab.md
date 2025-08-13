# Python-GitLab: Interact with GitLab APIs in Python

**Python-GitLab** is the go-to Python package for seamless interaction with GitLab's API, empowering developers to automate tasks and manage GitLab resources with ease.  [Explore the original repository](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic API Access:**  Write clean, readable Python code to manage your GitLab projects, users, and more.
*   **Comprehensive API Coverage:** Supports the v4 REST API, and synchronous & asynchronous GraphQL APIs.
*   **Flexible Parameter Handling:** Pass arbitrary parameters to the GitLab API based on GitLab's documentation.
*   **Asynchronous & Synchronous GraphQL Support:** Choose the best client for your needs.
*   **Access to Latest Endpoints:** Stay up-to-date with GitLab features via lower-level API methods.
*   **Persistent Sessions:** Leverage persistent requests sessions for improved authentication and proxy handling.
*   **Smart Error Handling:** Benefit from automatic retries, rate-limit handling and smart handling of network/server errors.
*   **Efficient Data Handling:** Handle paginated responses, including lazy iterators, for large datasets.
*   **Automatic Encoding:** URL-encode paths and parameters automatically for ease of use.
*   **Data Type Conversion:** Automatically convert complex data structures to API-compatible types.
*   **Configuration Flexibility:** Merge configuration from config files, environment variables, and arguments.
*   **CLI Tool:** Utilize the `gitlab` CLI tool to interact with REST API endpoints directly from your terminal.

## Installation

Python-GitLab requires Python 3.9 or later.

Install the latest stable version using pip:

```bash
pip install --upgrade python-gitlab
```

Or, install directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

Alternatively, install from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

Python-GitLab provides Docker images based on Alpine and Debian slim Python base images.

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

For example:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Using Inside GitLab CI:**

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

**Building Your Own Image:**

Build an Alpine-based image:

```bash
docker build -t python-gitlab:latest .
```

Run your built image:

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the [Gitter Community](https://gitter.im/python-gitlab/Lobby) for support and discussions.

## Documentation

Comprehensive documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.