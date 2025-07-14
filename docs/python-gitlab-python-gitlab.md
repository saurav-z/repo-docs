# Python-GitLab: Interact with GitLab APIs Seamlessly

**Python-GitLab is a powerful Python package that provides easy access and management of GitLab resources through its comprehensive APIs.**

[Visit the original repository on GitHub](https://github.com/python-gitlab/python-gitlab)

## Key Features:

*   **Pythonic Interface:** Write clean and readable Python code to manage your GitLab resources.
*   **Flexible API Access:**  Pass any parameters to the GitLab API, leveraging GitLab's documentation.
*   **GraphQL API Support:** Utilize both synchronous and asynchronous clients for the GraphQL API.
*   **Direct Endpoint Access:** Access any GitLab API endpoint as soon as it is available.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Robust Error Handling:** Includes smart retries for network and server errors, along with rate-limit handling.
*   **Paginated Response Handling:** Effortlessly handle paginated responses with lazy iterators.
*   **Automatic Data Encoding:** Automatically URL-encode paths and parameters.
*   **Data Type Conversion:** Automatically convert complex data structures to API attribute types.
*   **Configuration Flexibility:** Merge configurations from config files, environment variables, and arguments.
*   **CLI Tool:** Includes a command-line interface (``gitlab``) to interact with the REST API.

## Installation

Python-GitLab requires Python 3.9 or later.

Install the latest stable version using `pip`:

```bash
pip install --upgrade python-gitlab
```

You can also install from the GitHub repository:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

Or from GitLab:

```bash
pip install git+https://gitlab.com/python-gitlab/python-gitlab.git
```

## Using Docker Images

Python-GitLab offers Docker images based on Alpine and Debian slim Python base images.

Available tags:

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (latest, alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest slim-bullseye)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0` (alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-alpine`
*   `registry.gitlab.com/python-gitlab/python-gitlab:v3.2.0-slim-bullseye`

Run a container:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

Example:

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest project get --id gitlab-org/gitlab
```

Mount your configuration:

```bash
docker run -it --rm -v /path/to/python-gitlab.cfg:/etc/python-gitlab.cfg registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

### Usage in GitLab CI

Override the entrypoint:

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

Build the image:

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

## Reporting Issues

Report bugs and feature requests at [https://github.com/python-gitlab/python-gitlab/issues](https://github.com/python-gitlab/python-gitlab/issues).

## Community Support

Join the Gitter community chat: [https://gitter.im/python-gitlab/Lobby](https://gitter.im/python-gitlab/Lobby)

## Documentation

Refer to the comprehensive documentation for CLI and API details: [http://python-gitlab.readthedocs.org/en/stable/](http://python-gitlab.readthedocs.org/en/stable/)

## Contributing

Refer to the contributing guidelines:  [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst)