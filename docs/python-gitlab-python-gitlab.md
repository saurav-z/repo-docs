# python-gitlab: The Comprehensive Python Library for Interacting with GitLab APIs

**Effortlessly manage your GitLab resources with python-gitlab, a powerful Python package offering a robust client for the GitLab v4 REST API, synchronous and asynchronous GraphQL API clients, and a versatile CLI tool.**  You can find the original repository at [https://github.com/python-gitlab/python-gitlab](https://github.com/python-gitlab/python-gitlab).

## Key Features

*   **Pythonic Interface:** Write clean, readable Python code to interact with GitLab.
*   **Flexible API Access:** Pass any parameter supported by the GitLab API, with easy access to new endpoints as they are released.
*   **GraphQL Support:** Utilize both synchronous and asynchronous clients for GitLab's GraphQL API.
*   **Low-Level API Access:** Access arbitrary endpoints using lower-level API methods for maximum flexibility.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust Error Handling:** Enjoy smart retries on network and server errors, including rate-limit handling.
*   **Pagination Handling:** Seamlessly handle paginated responses with lazy iterators.
*   **Automated Encoding:** Automatically URL-encode paths and parameters.
*   **Data Conversion:** Automatically converts complex data structures to API attribute types.
*   **Configuration Management:**  Merge configurations from config files, environment variables, and command-line arguments.

## Installation

python-gitlab requires Python 3.9 or later.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or install the development version directly from GitHub or GitLab:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```
```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

python-gitlab offers Docker images based on Alpine and Debian slim Python base images.

*   **Alpine (default):** Smaller image.
*   **Debian Slim:** Provides a more complete environment and is better for CI jobs.

Images are published on the GitLab registry:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

**Run the Docker Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example (without authentication):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mount a config file:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

To use the Docker image inside GitLab CI, you will need to override the `entrypoint`:

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

### Building the image

**Build your own image:**

```bash
docker build -t python-gitlab:latest .
```

**Run your image:**

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

**Build a Debian slim-based image:**

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community Chat

Join the Gitter community chat for support and discussions: [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby).

## Documentation

Comprehensive documentation for both the CLI and API is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

### Build the Docs

```bash
pip install tox
tox -e docs
```

## Contributing

Refer to `CONTRIBUTING.rst <https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst>` for contribution guidelines.