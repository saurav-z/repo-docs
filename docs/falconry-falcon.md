![Falcon Logo](https://raw.githubusercontent.com/falconry/falcon/master/logo/banner.jpg)
_Building fast, reliable, and scalable REST APIs is easy with Falcon, a high-performance Python web framework._ ([Original Repo](https://github.com/falconry/falcon))

[![Build Status](https://github.com/falconry/falcon/actions/workflows/tests.yaml/badge.svg)](https://github.com/falconry/falcon/actions/workflows/tests.yaml)
[![Docs](https://readthedocs.org/projects/falcon/badge/?version=stable)](https://falcon.readthedocs.io/en/stable/?badge=stable)
[![codecov.io](https://codecov.io/gh/falconry/falcon/branch/master/graphs/badge.svg)](https://codecov.io/gh/falconry/falcon)
[![PyPI package](https://badge.fury.io/py/falcon.svg)](https://pypi.org/project/falcon/)
[![Python versions](https://img.shields.io/pypi/pyversions/falcon.svg)](https://pypi.org/project/falcon/)

## Falcon: The High-Performance Python Web Framework

Falcon is a minimalist ASGI/WSGI framework, tailor-made for building robust REST APIs and microservices. It's designed with a sharp focus on performance, reliability, and correctness at scale.

**Key Features:**

*   **ASGI, WSGI, and WebSocket Support:** Seamlessly integrates with various server environments.
*   **Native `asyncio` Support:** Leverage the power of asynchronous programming.
*   **Minimal Dependencies:** Reduces attack surface and avoids dependency conflicts.
*   **Clean API Design:** Embraces HTTP and REST principles.
*   **Simple API Modeling:**  RESTful routing for intuitive API design.
*   **Fast & Optimized:**  Provides excellent performance.
*   **Flexible:** Gives you control and customization options.
*   **Idiomatic HTTP Error Responses:**  Makes debugging and understanding API interactions easier.
*   **Supports Python 3.9+ and PyPy 3.9+:**  Compatible with modern Python environments.

### Quick Links
*   [Read the Docs](https://falcon.readthedocs.io/en/stable/) (FAQ, getting help, and API reference)
*   [Falcon Add-ons and Complementary Packages](https://github.com/falconry/falcon/wiki)
*   [Articles, Talks, and Podcasts](https://github.com/falconry/falcon/wiki/Articles,-Talks-and-Podcasts)
*   [Gitter Community for Users](https://gitter.im/falconry/user)
*   [Gitter Community for Developers](https://gitter.im/falconry/dev)

### What People Are Saying
> "Falcon is rock solid and it's fast."

> "We have been using Falcon as a replacement for [another framework] and we simply love the performance (three times faster) and code base size (easily half of our [original] code)."

> "I'm loving #falconframework! Super clean and simple, I finally have the speed and flexibility I need!"

### How is Falcon Different?

Falcon is built for the demanding needs of large-scale microservices and responsive application backends. It prioritizes reliability, debuggability, and speed.

*   **Reliable:**  Focuses on backwards-compatibility, rigorous testing, and minimal dependencies.
*   **Debuggable:** Eschews magic and provides clear input-output relationships.
*   **Fast:** Achieves significant performance gains compared to other popular Python frameworks.
*   **Flexible:** Gives developers control over implementation details, fostering customization.

### Who's Using Falcon?

Falcon powers applications for organizations worldwide, including:
* 7ideas
* Cronitor
* EMC
* Hurricane Electric
* Leadpages
* OpenStack
* Rackspace
* Shiftgig
* tempfil.es
* Opera Software

**Join the growing list of users!** If you use Falcon, add your project to our [wiki](https://github.com/falconry/falcon/wiki/Who's-using-Falcon%3F)

### Installation
#### PyPy
```bash
$ pip install falcon
```
To install the latest beta or release candidate:
```bash
$ pip install --pre falcon
```
#### CPython
```bash
$ pip install falcon
```
To install the latest beta or release candidate:
```bash
$ pip install --pre falcon
```
Falcon automatically compiles itself with `Cython <https://cython.org/>`__ for an extra speed boost. Pre-compiled binaries are available for many platforms on PyPI.

### Dependencies

Falcon has no dependencies.

### WSGI Server
Requires a WSGI server. Gunicorn and uWSGI are popular.
```bash
$ pip install [gunicorn|uwsgi]
```
### ASGI Server
Requires an ASGI server. Uvicorn is a popular choice.
```bash
$ pip install uvicorn
```
### Source Code
Find Falcon's source code on [GitHub](https://github.com/falconry/falcon).

To install from source:
```bash
$ cd falcon
$ pip install .
```
For development with symbolic linking:
```bash
$ cd falcon
$ FALCON_DISABLE_CYTHON=Y pip install -e .
```
### Testing
```bash
$ pip install -r requirements/tests
$ pytest tests
```
Or to run the default set of tests:
```bash
$ pip install tox && tox
```
### Documentation

Comprehensive documentation is available at: https://falcon.readthedocs.io

Build docs locally:

```bash
$ pip install tox && tox -e docs
```

Open the built docs in your browser:
```bash
$ open docs/_build/html/index.html # OS X
$ xdg-open docs/_build/html/index.html # Linux
```

### Getting Started

[Simple WSGI Example](https://github.com/falconry/falcon#getting-started)

[Simple ASGI Example](https://github.com/falconry/falcon#getting-started)

[More Complex Examples (WSGI and ASGI)](https://github.com/falconry/falcon#a-more-complex-example-wsgi)

### Contributing

We welcome contributions!  See our [CONTRIBUTING.md](https://github.com/falconry/falcon/blob/master/CONTRIBUTING.md) and [Code of Conduct](https://github.com/falconry/falcon/blob/master/CODEOFCONDUCT.md).

### Legal
Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:** The title emphasizes the framework's name, and the first sentence acts as an attention-grabbing hook describing its purpose.
*   **SEO-Friendly Keywords:**  Includes keywords like "Python," "web framework," "REST API," "microservices," "ASGI," "WSGI," "performance," and "scalable."
*   **Structured Headings:** Uses clear headings (e.g., "Key Features," "Getting Started") to improve readability and SEO.
*   **Bulleted Lists:**  Uses bullet points for easy scanning of key features.
*   **Concise Descriptions:** Keeps descriptions short and to the point.
*   **Links to Documentation:**  Prominently features links to the official documentation, ensuring discoverability.
*   **Community Section:** Highlights community resources (Gitter) to encourage engagement.
*   **Installation Guide:**  More direct and easy-to-follow installation steps.
*   **Call to Action:** Encourages users to add their project to the "Who's Using Falcon?" list.
*   **Concise "How is Falcon Different?" Section:**  Summarizes key differentiating aspects like reliability, debuggability, speed, and flexibility.
*   **Emphasis on Performance:** The descriptions emphasize Falcon's speed and performance benefits.
*   **Link to Original Repo:** Maintains a clear link back to the source repository at the top.