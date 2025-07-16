# python-gitlab: Python Client for the GitLab API

**Effortlessly interact with GitLab through Python using `python-gitlab`, a robust and feature-rich library.** Explore the [python-gitlab GitHub repository](https://github.com/python-gitlab/python-gitlab) for more details and contribute to its development.

## Key Features

*   **Pythonic API:** Write clean and readable Python code to manage your GitLab resources.
*   **Flexible API Access:** Interact with any GitLab API endpoint, including the v4 REST API and both synchronous and asynchronous GraphQL APIs.
*   **Asynchronous GraphQL Support:** Utilize both synchronous and asynchronous clients when using the GraphQL API.
*   **Arbitrary Parameter Passing:** Pass any parameters to the GitLab API, following GitLab's documentation.
*   **Persistent Sessions:** Leverage persistent request sessions for authentication, proxy, and certificate handling.
*   **Smart Error Handling:** Benefit from smart retries on network and server errors, including rate-limit handling.
*   **Pagination Support:** Handle paginated responses efficiently, including lazy iterators.
*   **Automatic Encoding:**  Paths and parameters are automatically URL-encoded.
*   **Data Conversion:** Complex data structures are automatically converted to API attribute types.
*   **Configuration Merging:** Configuration can be merged from config files, environment variables, and arguments.
*   **CLI Tool:** Includes a command-line interface (CLI) tool (`gitlab`) for easy interaction with the REST API.

## Installation

`python-gitlab` requires Python 3.9 or later.

Install using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** `registry.gitlab.com/python-gitlab/python-gitlab:latest` or `registry.gitlab.com/python-gitlab/python-gitlab:alpine`
*   **Debian slim:** `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye`

Run a Docker image (example):

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

## Usage inside GitLab CI

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

## Build Your Own Image

Build a Docker image locally:

```bash
docker build -t python-gitlab:latest .
docker run -it --rm python-gitlab:latest <command> ...
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Get Help

*   **Bug Reports and Feature Requests:** Submit issues on [GitHub](https://github.com/python-gitlab/python-gitlab/issues).
*   **Community Chat:** Join the [Gitter community](https://gitter.im/python-gitlab/Lobby) for discussions and support.
*   **Documentation:**  Explore the full documentation on [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to `CONTRIBUTING.rst <https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst>` for guidelines on contributing to `python-gitlab`.