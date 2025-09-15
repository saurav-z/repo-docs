[![Falcon Logo](https://raw.githubusercontent.com/falconry/falcon/master/logo/banner.jpg)](https://falconframework.org/)

# Falcon: Build blazing-fast REST APIs and Microservices

Falcon is a high-performance, minimalist Python web framework designed for building reliable, scalable REST APIs and microservices, offering speed and flexibility for demanding applications.  [Visit the Falcon GitHub Repository](https://github.com/falconry/falcon).

[![Build Status](https://github.com/falconry/falcon/actions/workflows/tests.yaml/badge.svg)](https://github.com/falconry/falcon/actions/workflows/tests.yaml)
[![Docs](https://readthedocs.org/projects/falcon/badge/?version=stable)](https://falcon.readthedocs.io/en/stable/?badge=stable)
[![Codecov](https://codecov.io/gh/falconry/falcon/branch/master/graphs/badge.svg)](https://codecov.io/gh/falconry/falcon)
[![PyPI package](https://badge.fury.io/py/falcon.svg)](https://pypi.org/project/falcon/)
[![Python versions](https://img.shields.io/pypi/pyversions/falcon.svg)](https://pypi.org/project/falcon/)

## Key Features

*   **ASGI, WSGI, and WebSocket support:**  Works with various servers for flexibility.
*   **Native asyncio support:**  For highly concurrent applications.
*   **No magic globals:**  Clear and predictable state management.
*   **Focus on backward compatibility:**  Stable and reliable interfaces.
*   **RESTful routing:**  Simple API design based on HTTP principles.
*   **Optimized code base:**  Designed for performance and scalability.
*   **Request/Response classes:**  Easy access to headers and bodies.
*   **Middleware & Hooks:**  DRY request processing.
*   **Strict RFC adherence:**  Ensures proper HTTP behavior.
*   **Idiomatic error handling:**  Provides clear error responses.
*   **Straightforward exception handling:**  Easy to debug and maintain.
*   **Testing tools:**  Convenient WSGI/ASGI helpers and mocks.
*   **Python 3.9+ and PyPy 3.9+ support:**  Wide compatibility for different environments.

## Getting Started

Falcon is designed to be easy to learn and use.  Here's a simple example:

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

Run this example with:

```bash
$ pip install falcon
$ python things.py
```

Then test it in another terminal:

```bash
$ curl localhost:8000/things
```

See the full documentation for more advanced examples.

## How is Falcon Different?

Falcon's design prioritizes performance, reliability, and flexibility:

*   **Reliable:**  Emphasizes backward compatibility and rigorous testing.
*   **Debuggable:**  Avoids "magic" and simplifies code paths for easy debugging.
*   **Fast:**  Significantly faster than some popular Python frameworks.
*   **Flexible:**  Gives developers control over implementation details.

## Installation

### PyPy

```bash
$ pip install falcon
```

Or, for the latest beta/release candidate:

```bash
$ pip install --pre falcon
```

### CPython

```bash
$ pip install falcon
```

Or, for the latest beta/release candidate:

```bash
$ pip install --pre falcon
```

## Learn More

*   [Read the Docs](https://falcon.readthedocs.io/en/stable/)
*   [FAQ](https://falcon.readthedocs.io/en/stable/user/faq.html)
*   [Falcon add-ons and complementary packages](https://github.com/falconry/falcon/wiki)
*   [Falcon articles, talks and podcasts](https://github.com/falconry/falcon/wiki/Articles,-Talks-and-Podcasts)
*   [Community](https://gitter.im/falconry/user)

##  Support Falcon Development

Has Falcon helped you make an awesome app? Show your support today with a
one-time donation or by becoming a patron.
Supporters get cool gear, an opportunity to promote their brand to Python
developers, and prioritized support.

*   [Learn how to support Falcon development](https://falconframework.org/#sectionSupportFalconDevelopment)

## Who's Using Falcon?

Falcon is used around the world by a growing number of organizations,
including:

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

If you are using the Falcon framework for a community or commercial
project, please consider adding your information to our wiki under
`Who's Using Falcon? <https://github.com/falconry/falcon/wiki/Who's-using-Falcon%3F>`_

## Contributing

We welcome contributions! Check out the [CONTRIBUTING.md](https://github.com/falconry/falcon/blob/master/CONTRIBUTING.md) for details.

### Core Maintainers:

*   Kurt Griffiths, Project Lead (**kgriffs**)
*   John Vrbanac (**jmvrbanac**)
*   Vytautas Liuolia (**vytas7**)
*   Nick Zaccardi (**nZac**)
*   Federico Caselli (**CaselIT**)

## Legal

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.