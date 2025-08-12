# python-gitlab: Interact with GitLab APIs in Python

**Seamlessly manage your GitLab projects and resources with the powerful and versatile `python-gitlab` library.**

[View the original repository on GitHub](https://github.com/python-gitlab/python-gitlab)

## Key Features

*   **Pythonic GitLab API Access:** Write clean, readable Python code to interact with GitLab.
*   **REST & GraphQL API Support:** Utilize both synchronous and asynchronous clients for GitLab's v4 REST and GraphQL APIs.
*   **Flexible API Parameter Passing:** Easily pass any parameters to the GitLab API, as documented by GitLab.
*   **GraphQL Client:** Use a synchronous or asynchronous client when using the GraphQL API.
*   **Comprehensive Coverage:** Access any GitLab endpoint, even new ones, using lower-level API methods.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust Error Handling:** Leverage smart retries for network and server errors, including rate-limit handling.
*   **Efficient Pagination:** Handle paginated responses effectively, including lazy iterators.
*   **Automatic Data Handling:** Automatically URL-encode paths and parameters, and convert complex data structures for API compatibility.
*   **Configuration Management:** Merge configuration from config files, environment variables, and command-line arguments.

## Installation

`python-gitlab` requires Python 3.9 or later.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

Or install the latest development version directly from GitHub or GitLab:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```
```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images, available on the GitLab registry.

**Available Images:**

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

**Running the Image:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Example (Get a project on GitLab.com):**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

**Mounting a Configuration File:**

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage Inside GitLab CI

When using the Docker image in GitLab CI, override the `entrypoint` as described in the [official GitLab documentation](https://docs.gitlab.com/ee/ci/docker/using_docker_images.html#override-the-entrypoint-of-an-image).

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

Build your own Docker image:

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

## Bug Reports

Report bugs and feature requests on [GitHub](https://github.com/python-gitlab/python-gitlab/issues).

## Gitter Community Chat

Join the `python-gitlab` community on [Gitter](https://gitter.im/python-gitlab/Lobby) for discussions and support.

## Documentation

Comprehensive documentation for both CLI and API is available on [readthedocs](http://python-gitlab.readthedocs.org/en/stable/).

## Contributing

Refer to `CONTRIBUTING.rst` on [GitHub](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst) for contribution guidelines.