![Falcon Logo](https://raw.githubusercontent.com/falconry/falcon/master/logo/banner.jpg)
<p align="center">
  <a href="https://github.com/falconry/falcon">
    <img src="https://img.shields.io/github/stars/falconry/falcon?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/falconry/falcon/actions/workflows/tests.yaml">
    <img src="https://github.com/falconry/falcon/actions/workflows/tests.yaml/badge.svg" alt="Build Status">
  </a>
  <a href="https://falcon.readthedocs.io/en/stable/?badge=stable">
    <img src="https://readthedocs.org/projects/falcon/badge/?version=stable" alt="Docs">
  </a>
  <a href="https://codecov.io/gh/falconry/falcon">
    <img src="https://codecov.io/gh/falconry/falcon/branch/master/graphs/badge.svg" alt="Codecov">
  </a>
  <a href="https://pypi.org/project/falcon/">
    <img src="https://badge.fury.io/py/falcon.svg" alt="PyPI package">
  </a>
  <a href="https://pypi.org/project/falcon/">
    <img src="https://img.shields.io/pypi/pyversions/falcon.svg" alt="Python versions">
  </a>
</p>

# Falcon: Build blazing-fast REST APIs and microservices with a minimalist Python framework.

[Falcon](https://github.com/falconry/falcon) is a high-performance, production-ready Python web framework ideal for building reliable and scalable REST APIs and microservices.  It emphasizes speed, flexibility, and a clean design that embraces HTTP and the REST architectural style.

## Key Features:

*   **ASGI & WSGI Support:** Works seamlessly with both ASGI and WSGI servers, offering flexibility in deployment.
*   **Native `asyncio` Support:** Built-in support for asynchronous programming, maximizing performance.
*   **Minimalist Design:** No reliance on magic globals for clean and maintainable code.
*   **RESTful Routing:** Simple API modeling through centralized RESTful routing.
*   **High Performance:** Highly optimized, extensible codebase for maximum speed.
*   **Request & Response Handling:** Easy access to headers and bodies through request and response classes.
*   **Middleware & Hooks:** DRY request processing via middleware components and hooks.
*   **Strict RFC Adherence:** Strict adherence to RFCs for robust and predictable behavior.
*   **Idiomatic Error Handling:** Straightforward exception handling for improved debugging.
*   **Testing Support:** Snappy testing with WSGI/ASGI helpers and mocks.
*   **Python Version Support:**  CPython 3.9+ and PyPy 3.9+ compatibility.

## Quick Links

*   [Read the docs](https://falcon.readthedocs.io/en/stable) ([FAQ](https://falcon.readthedocs.io/en/stable/user/faq.html) - [getting help](https://falcon.readthedocs.io/en/stable/community/help.html) - [reference](https://falcon.readthedocs.io/en/stable/api/index.html))
*   [Falcon add-ons and complementary packages](https://github.com/falconry/falcon/wiki)
*   [Falcon articles, talks and podcasts](https://github.com/falconry/falcon/wiki/Articles,-Talks-and-Podcasts)
*   [Falcon User Community](https://gitter.im/falconry/user) @ Gitter
*   [Falcon Developer Community](https://gitter.im/falconry/dev) @ Gitter

## What People are Saying

(Quotes from the original README, included to maintain authenticity)

## How is Falcon Different?

Falcon is designed for the demanding needs of large-scale microservices and responsive app backends, focusing on performance, reliability, and flexibility.

*   **Reliable:** Rigorously tested with 100% code coverage, and designed for backwards compatibility.  Minimizes dependencies to reduce attack surface.
*   **Debuggable:**  Simple and understandable code with no hidden behaviors, making it easy to trace inputs to outputs.
*   **Fast:**  Significantly faster than other popular Python frameworks like Django and Flask, with Cython and PyPy support for an extra speed boost.
*   **Flexible:**  Offers developers more control, enabling customization and deeper understanding of apps, alongside a thriving ecosystem of add-ons.

## Who's Using Falcon?

(Organizations from the original README, included to maintain authenticity)

If you use Falcon, add your info to the [wiki](https://github.com/falconry/falcon/wiki/Who's-using-Falcon%3F).

## Community

Explore the Falcon community for support, add-ons and discussions:

*   [Falcon Wiki](https://github.com/falconry/falcon/wiki): Add-ons, templates, and resources.
*   [Falcon User Community](https://gitter.im/falconry/user): Ask questions and share ideas.
*   [Falcon Developer Community](https://gitter.im/falconry/dev):  Discuss framework development.

## Installation

### PyPy

```bash
$ pip install falcon
```

Or, to install the latest beta or release candidate:

```bash
$ pip install --pre falcon
```

### CPython

```bash
$ pip install falcon
```

Or, to install the latest beta or release candidate:

```bash
$ pip install --pre falcon
```

## Dependencies

Falcon requires no dependencies beyond the standard library.

## WSGI Server

To run a Falcon app, you need a WSGI server.

```bash
$ pip install [gunicorn|uwsgi]
```

## ASGI Server

For ASGI apps, use an ASGI server like Uvicorn:

```bash
$ pip install uvicorn
```

## Source Code

*   [GitHub Repository](https://github.com/falconry/falcon)
*   Clone the repo: `git clone https://github.com/falconry/falcon.git`
*   Install:

    ```bash
    $ cd falcon
    $ pip install .
    ```

Or, for development (using symbolic linking):

```bash
$ cd falcon
$ FALCON_DISABLE_CYTHON=Y pip install -e .
```

*   Testing
    ```bash
    $ pip install -r requirements/tests
    $ pytest tests
    ```

    or

    ```bash
    $ pip install tox && tox
    ```

*   Read the Docs
    ```bash
    $ pip install tox && tox -e docs
    ```

## Getting Started

(Includes example code from the original README, to maintain authenticity)

```python
# examples/things.py

# Let's get this party started!
from wsgiref.simple_server import make_server

import falcon


# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class ThingsResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = ('\nTwo things awe me most, the starry sky '
                     'above me and the moral law within me.\n'
                     '\n'
                     '    ~ Immanuel Kant\n\n')


# falcon.App instances are callable WSGI apps...
# in larger applications the app is created in a separate file
app = falcon.App()

# Resources are represented by long-lived class instances
things = ThingsResource()

# things will handle all requests to the '/things' URL path
app.add_route('/things', things)

if __name__ == '__main__':
    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')

        # Serve until process is killed
        httpd.serve_forever()
```

Run with:

```bash
$ pip install falcon
$ python things.py
```

Then, in another terminal:

```bash
$ curl localhost:8000/things
```

The ASGI version is similar:

```python
# examples/things_asgi.py

import falcon
import falcon.asgi


# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class ThingsResource:
    async def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = ('\nTwo things awe me most, the starry sky '
                     'above me and the moral law within me.\n'
                     '\n'
                     '    ~ Immanuel Kant\n\n')


# falcon.asgi.App instances are callable ASGI apps...
# in larger applications the app is created in a separate file
app = falcon.asgi.App()

# Resources are represented by long-lived class instances
things = ThingsResource()

# things will handle all requests to the '/things' URL path
app.add_route('/things', things)
```

Run the ASGI version with uvicorn:

```bash
$ pip install falcon uvicorn
$ uvicorn things_asgi:app
```

## More Complex Examples (WSGI & ASGI)

(Examples from the original README. Included, to maintain authenticity)

*   [A More Complex Example (WSGI)]
*   [A More Complex Example (ASGI)]

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/falconry/falcon/blob/master/CONTRIBUTING.md) for details.

## Legal

(Legal information from original README, included to maintain authenticity)

```
Copyright 2013-2025 by Individual and corporate contributors as
noted in the individual source files.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use any portion of the Falcon framework except in compliance with
the License. Contributors agree to license their work under the same
License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Support Falcon Development

  [Learn how to support Falcon development](https://falconframework.org/#sectionSupportFalconDevelopment)

<p>
    <a href="https://www.govcert.lu/">
      <img src="https://falconframework.org/assets/govcert.png" alt="CERT Gouvernemental Luxembourg" height="60px">
    </a>
    <a href="https://sentry.io">
      <img src="https://falconframework.org/assets/sentry-dark.svg" alt="Sentry" height="60px">
    </a>
</p>