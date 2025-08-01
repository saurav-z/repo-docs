# python-gitlab: Python Library for the GitLab API

**Effortlessly interact with your GitLab projects and resources using the python-gitlab library, providing both REST and GraphQL API clients.** You can find the original repository [here](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Comprehensive API Access:** Interact with GitLab's v4 REST API and GraphQL API.
*   **Synchronous & Asynchronous GraphQL Clients:** Choose the client type that best suits your needs.
*   **CLI Tool:** Utilize the `gitlab` CLI for convenient access to REST API endpoints.
*   **Pythonic Code:** Write clean and readable Python code to manage your GitLab resources.
*   **Flexible Parameter Passing:** Pass any parameters supported by the GitLab API directly.
*   **Lower-Level API Access:** Access new endpoints as soon as they become available in GitLab.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxies, and certificate handling.
*   **Smart Retries:** Handle network and server errors with smart retries and rate-limit handling.
*   **Paginated Response Handling:** Effortlessly navigate paginated responses with lazy iterators.
*   **Automatic Data Handling:** URL-encode paths and parameters and convert complex data structures to API attribute types automatically.
*   **Configuration Management:** Merge configuration from config files, environment variables, and arguments.

## Installation

``python-gitlab`` is compatible with Python 3.9+.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or, install the latest development version from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

``python-gitlab`` provides Docker images based on Alpine and Debian slim python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest`
*   **Debian slim-bullseye:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

**Run a Docker image example:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

## Usage Inside GitLab CI

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

## Bug Reports

Report bugs and feature requests at https://github.com/python-gitlab/python-gitlab/issues.

## Community Chat

Join the Gitter community at https://gitter.im/python-gitlab/Lobby for discussions and support.

## Documentation

The full documentation is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contributing guidelines.