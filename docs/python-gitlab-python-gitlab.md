# Python-GitLab: A Powerful Python Library for Interacting with the GitLab API

**Python-GitLab** is a robust Python package designed to simplify interaction with the GitLab API, offering a comprehensive solution for automating and managing your GitLab resources. Explore the [original repository](https://github.com/python-gitlab/python-gitlab) for more information and to contribute.

## Key Features

*   **Pythonic API:** Write clean, readable Python code to manage your GitLab projects, users, groups, and more.
*   **Flexible API Access:** Pass any parameters supported by the GitLab API directly through the library, staying up-to-date with new features.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for interacting with the GitLab GraphQL API.
*   **Low-Level API Access:** Access any GitLab API endpoint, even those newly released, using lower-level API methods.
*   **Persistent Sessions:** Benefit from persistent request sessions for efficient authentication and proxy/certificate handling.
*   **Robust Error Handling:** Enjoy smart retries for network and server errors, with built-in rate-limit handling to prevent issues.
*   **Pagination and Iteration:** Easily handle paginated responses with flexible options, including lazy iterators for efficient data retrieval.
*   **Automatic Data Handling:** Automatically URL-encode paths and parameters and convert complex data structures to appropriate API attribute types.
*   **Configuration Flexibility:** Merge configuration settings seamlessly from configuration files, environment variables, and command-line arguments.

## Installation

Python-GitLab is compatible with Python 3.9 and later.

Use `pip` to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Alternatively, install the current development version directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

Python-GitLab provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine:**  `registry.gitlab.com/python-gitlab/python-gitlab:latest` (default)
*   **Debian slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (recommended for more complete environments)

**Running the Docker Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example:** Get a project on GitLab.com:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Config File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Usage inside GitLab CI:**  Override the `entrypoint` in your `.gitlab-ci.yml` file:

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

**Building the Docker Image:**

```bash
docker build -t python-gitlab:latest .
```

**Building a Debian slim-based image:**

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports and Community

*   **Report Bugs & Feature Requests:**  https://github.com/python-gitlab/python-gitlab/issues
*   **Gitter Community Chat:** https://gitter.im/python-gitlab/Lobby

## Documentation

Comprehensive documentation is available on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to `CONTRIBUTING.rst` for guidelines: https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst