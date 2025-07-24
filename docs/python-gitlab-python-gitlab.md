# python-gitlab: Interact with GitLab APIs Effortlessly in Python

**python-gitlab** is a powerful Python package that provides a robust and easy-to-use interface for interacting with the GitLab APIs.

[Go to the original repository](https://github.com/python-gitlab/python-gitlab)

## Key Features

*   **Pythonic Code:** Write clean, readable Python code to manage GitLab resources.
*   **Full API Coverage:** Access all GitLab API endpoints, including REST and GraphQL APIs.
*   **Asynchronous and Synchronous Clients:**  Utilize both synchronous and asynchronous clients for the GraphQL API, for optimal performance.
*   **Arbitrary Parameter Support:** Easily pass any parameters supported by the GitLab API directly through your code.
*   **Flexible Configuration:** Merge configurations from config files, environment variables, and arguments.
*   **Persistent Sessions:** Maintain persistent request sessions for authentication, proxy, and certificate handling.
*   **Smart Error Handling:** Benefit from automatic retries on network and server errors, with built-in rate-limit handling.
*   **Paginated Response Handling:** Seamlessly handle paginated responses with lazy iterators for efficient data retrieval.
*   **Automatic Encoding:** Enjoy automatic URL-encoding of paths and parameters.
*   **Data Conversion:** Automatically convert complex data structures to API attribute types.

## Installation

**Requirements:** `python-gitlab` is compatible with Python 3.9+.

Use `pip` to install the latest stable version:

```bash
pip install --upgrade python-gitlab
```

Install the development version from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

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

**Example (Get Project):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Config File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

When using the Docker image in GitLab CI, you may need to override the `entrypoint`:

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

```bash
docker build -t python-gitlab:latest .
```

**Building a Debian Slim-based Image:**

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the Gitter community chat for help and discussion at [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby).

## Documentation

Detailed documentation for both the CLI and API is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

### Build the Docs

```bash
pip install tox
tox -e docs
```

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.