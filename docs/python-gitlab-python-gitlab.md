# python-gitlab: The Python Library for Seamless GitLab API Interaction

Easily manage your GitLab projects and resources with the robust and versatile `python-gitlab` library.  [Check out the original repository](https://github.com/python-gitlab/python-gitlab) for more information.

## Key Features

*   **Pythonic API Access:** Interact with GitLab using intuitive Python code.
*   **Comprehensive API Coverage:** Access both REST and GraphQL APIs.
*   **GraphQL Support:** Utilize both synchronous and asynchronous GraphQL clients.
*   **CLI Tool:** Leverage a command-line interface (`gitlab`) for easy API interaction.
*   **Arbitrary Parameter Passing:** Pass any parameters to the GitLab API, as documented by GitLab.
*   **Persistent Sessions:** Benefit from persistent request sessions for authentication, proxy, and certificate handling.
*   **Smart Retries:** Experience automatic retries for network and server errors, with rate-limit handling.
*   **Paginated Response Handling:** Easily navigate through paginated responses with flexible options, including lazy iterators.
*   **Automatic Encoding:** Path and parameter encoding is handled automatically.
*   **Data Conversion:** Some complex data structures are automatically converted to API attribute types.
*   **Configuration Flexibility:** Merge configurations from config files, environment variables, and arguments.

## Installation

`python-gitlab` is compatible with Python 3.9+. Install the latest stable version using pip:

```bash
pip install --upgrade python-gitlab
```

You can also install the development version directly from GitHub:

```bash
pip install git+https://github.com/python-gitlab/python-gitlab.git
```

## Docker Images

`python-gitlab` provides Docker images based on Alpine and Debian slim Python base images, published on the GitLab registry.

*   `registry.gitlab.com/python-gitlab/python-gitlab:latest` (Alpine alias)
*   `registry.gitlab.com/python-gitlab/python-gitlab:alpine` (latest Alpine)
*   `registry.gitlab.com/python-gitlab/python-gitlab:slim-bullseye` (latest Debian slim)
*   And more...

**Running a Docker Container:**

```bash
docker run -it --rm registry.gitlab.com/python-gitlab/python-gitlab:latest <command> ...
```

**Using with GitLab CI:**

Override the `entrypoint` in your GitLab CI configuration:

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

## Resources

*   **Bug Reports & Feature Requests:** [GitHub Issues](https://github.com/python-gitlab/python-gitlab/issues)
*   **Community Chat:** [Gitter](https://gitter.im/python-gitlab/Lobby)
*   **Documentation:** [Read the Docs](http://python-gitlab.readthedocs.org/en/stable/)
*   **Contributing:** [CONTRIBUTING.rst](https://github.com/python-gitlab/python-gitlab/blob/main/CONTRIBUTING.rst)