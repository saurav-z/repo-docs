[![Falcon Logo](https://raw.githubusercontent.com/falconry/falcon/master/logo/banner.jpg)](https://falconframework.org/)

# Falcon: Build Fast, Reliable REST APIs with Python

**Falcon is a minimalist ASGI/WSGI framework designed for high-performance REST APIs and microservices.**  Get the edge in speed and reliability for your mission-critical applications.

[View the original repository on GitHub](https://github.com/falconry/falcon)

**Key Features:**

*   **ASGI, WSGI, and WebSocket Support:** Compatible with modern Python web server interfaces for flexibility.
*   **Native `asyncio` Support:** Built-in asynchronous capabilities for efficient handling of concurrent requests.
*   **No Magic Globals:**  Clean design with explicit routing and state management.
*   **Stable & Backwards-Compatible Interfaces:** Focus on reliability and ease of upgrades.
*   **RESTful Routing:**  Centralized, intuitive API design.
*   **Highly-Optimized & Extensible:**  Fast performance with a customizable codebase.
*   **Request/Response Classes:**  Easy access to headers and bodies.
*   **Middleware and Hooks:**  Simplified request processing.
*   **Strict RFC Adherence:**  Complies with HTTP standards.
*   **Idiomatic Error Handling:**  Handles errors gracefully.
*   **Simple Testing:**  Uses WSGI/ASGI helpers and mocks for robust testing.
*   **CPython & PyPy Support:** Runs on CPython 3.9+ and PyPy 3.9+.

**Quick Links:**

*   [Read the Docs](https://falcon.readthedocs.io/en/stable) (FAQ, getting help, reference)
*   [Falcon Add-ons & Packages](https://github.com/falconry/falcon/wiki)
*   [Articles, Talks & Podcasts](https://github.com/falconry/falcon/wiki/Articles,-Talks-and-Podcasts)
*   [Falcon Users on Gitter](https://gitter.im/falconry/user)
*   [Falcon Dev on Gitter](https://gitter.im/falconry/dev)

**What People Are Saying:**

> "Falcon is rock solid and it's fast."
>
> "We have been using Falcon as a replacement for [another framework] and we simply love the performance (three times faster) and code base size (easily half of our [original] code)."
>
> "I'm loving #falconframework! Super clean and simple, I finally have the speed and flexibility I need!"
>
> "...was ~40% faster with only 20 minutes of work."
>
> "I feel like I'm just talking HTTP at last, with nothing in the middle. Falcon seems like the requests of backend."
>
> "The source code for Falcon is so good, I almost prefer it to documentation. It basically can't be wrong."
>
> "What other framework has integrated support for 786 TRY IT NOW ?"

**How is Falcon Different?**

Falcon is designed for:

*   **Reliability:** Focus on avoiding breaking changes and rigorous testing.
*   **Debuggability:**  Clear code and easy-to-follow logic paths.
*   **Speed:** Significantly faster than other popular Python frameworks.
*   **Flexibility:** Allows you to customize your implementation.

**Who's Using Falcon?**

Falcon is used by organizations like:

*   7ideas
*   Cronitor
*   EMC
*   Hurricane Electric
*   Leadpages
*   OpenStack
*   Rackspace
*   Shiftgig
*   tempfil.es
*   Opera Software

[Add your project to the Falcon wiki](https://github.com/falconry/falcon/wiki/Who's-using-Falcon%3F)

**Community:**

A vibrant community provides Falcon add-ons and resources.

*   [Falcon Wiki](https://github.com/falconry/falcon/wiki)
*   [Falcon Users on Gitter](https://gitter.im/falconry/user)
*   [Falcon Dev on Gitter](https://gitter.im/falconry/dev)

**Installation:**

*   **PyPy:**  The fastest way to run your Falcon app.
    ```bash
    $ pip install falcon
    ```
*   **CPython 3.9+:**
    ```bash
    $ pip install falcon
    ```
    (For the latest versions: `pip install --pre falcon`)

**Dependencies:**

Falcon has no external dependencies.

**WSGI Server:**

You'll need a WSGI server to serve your Falcon app:
```bash
$ pip install [gunicorn|uwsgi]
```

**ASGI Server:**

To serve a Falcon ASGI app, use an ASGI server:
```bash
$ pip install uvicorn
```

**Source Code:**

*   [GitHub Repository](https://github.com/falconry/falcon)
*   Clone the repo: `git clone https://github.com/falconry/falcon.git`
*   Install locally: `pip install .`
*   Edit the code: `FALCON_DISABLE_CYTHON=Y pip install -e .`
*   Run tests: `pytest tests` or `pip install tox && tox`

**Read the Docs:**

*   [Online Documentation](https://falcon.readthedocs.io)
*   Build locally: `pip install tox && tox -e docs`

**Getting Started:**

See the [Getting Started section](https://github.com/falconry/falcon#getting-started) in the full README.

**Contributing:**

We welcome your contributions.

*   [Contributing Guidelines](https://github.com/falconry/falcon/blob/master/CONTRIBUTING.md)

**Legal:**

*   [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0)

---