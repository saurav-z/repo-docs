# Python-GitLab: Your Gateway to GitLab APIs

**Python-GitLab is a powerful Python package that simplifies interacting with GitLab APIs, empowering developers to manage their GitLab resources programmatically.** Learn more and contribute to the project on [GitHub](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic Code:** Write clean, readable Python code to interact with GitLab.
*   **API Flexibility:** Pass any parameters to the GitLab API as per GitLab's documentation.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for the GraphQL API.
*   **Access to All Endpoints:** Access any GitLab endpoint as soon as it's available, using lower-level API methods.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Smart Retries & Rate Limiting:** Includes smart retries for network and server errors with rate-limit handling.
*   **Paginated Response Handling:** Flexible handling of paginated responses, including lazy iterators.
*   **Automatic Encoding:** Automatically URL-encode paths and parameters.
*   **Data Conversion:** Automatically convert complex data structures to API attribute types.
*   **Configuration Management:** Merge configuration from config files, environment variables, and arguments.

## Installation

Python-GitLab requires Python 3.9 or higher.

Use `pip` to install the latest stable version:

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

Python-GitLab provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest`
*   **Debian Slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (recommended if you need bash shell)

To run a Docker image:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Example:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

You can also mount your own config file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Using in GitLab CI

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

### Building the Image

To build your own image:

```bash
docker build -t python-gitlab:latest .
```

To build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports & Feature Requests

Report issues at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the Gitter community for questions and discussions: [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby).

## Documentation

Comprehensive documentation is available on Read the Docs: [http://python-gitlab.readthedocs.org/en/stable/](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contributing guidelines.