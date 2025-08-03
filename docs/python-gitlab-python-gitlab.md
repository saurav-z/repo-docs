# python-gitlab: Interact with GitLab APIs using Python

**Easily manage your GitLab resources with `python-gitlab`, a powerful Python package providing access to the GitLab APIs.**

[View the original repository on GitHub](https://github.com/python-gitlab/python-gitlab)

Key Features:

*   **Pythonic Interface:** Write clean, readable Python code for GitLab interactions.
*   **Flexible API Access:** Pass custom parameters to GitLab APIs, leveraging the full power of GitLab's documentation.
*   **GraphQL Support:** Use synchronous or asynchronous clients for the GraphQL API.
*   **Comprehensive API Coverage:** Access all GitLab endpoints, including lower-level methods for immediate availability of new features.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust Error Handling:** Automatic smart retries for network and server errors, including rate-limit handling.
*   **Pagination & Iteration:** Handle paginated responses with ease, including lazy iterators.
*   **Automated Encoding:** Automatically URL-encode paths and parameters as needed.
*   **Data Conversion:** Automatically convert complex data structures to API attribute types.
*   **Configuration Management:** Merge configuration from config files, environment variables, and command-line arguments.

## Installation

`python-gitlab` is compatible with Python 3.9+.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

You can also install directly from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

or from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using the Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images.

Available images on the GitLab registry:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

Run the Docker image:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

For example:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

Mount your config file:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage inside GitLab CI

Override the `entrypoint` for the image:

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

Build your own image:

```bash
docker build -t python-gitlab:latest .
```

Run your image:

```bash
docker run -it --rm python-gitlab:latest <command> ...
```

Build a Debian slim-based image:

```bash
docker build -t python-gitlab:latest --build-arg PYTHON_FLAVOR=slim-bullseye .
```

## Bug Reports

Report bugs and feature requests at [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community

Join the [Gitter community chat](https://gitter.im/python-gitlab/Lobby) to ask questions and discuss ideas.

## Documentation

Read the full documentation on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

See [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for guidelines.