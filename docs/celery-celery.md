<!-- ALL-IN-ONE SEO META DESCRIPTION -->
# Celery: Asynchronous Task Queue for Python

**Celery is a powerful and flexible distributed task queue that enables you to run tasks asynchronously, in the background, or on different machines.** [See the original repository](https://github.com/celery/celery).

[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage Status](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementations](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)
[![Open Collective Backers](https://opencollective.com/celery/backers/badge.svg)](https://opencollective.com/celery#backers)
[![Open Collective Sponsors](https://opencollective.com/celery/sponsors/badge.svg)](https://opencollective.com/celery#sponsors)
[![Downloads](https://pepy.tech/badge/celery)](https://pepy.tech/project/celery)
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/celery/celery)
<!-- /ALL-IN-ONE SEO META DESCRIPTION -->

## Key Features

*   **Simple:** Easy to use and maintain without configuration files.
*   **Highly Available:** Automatic retries for workers and clients, with broker HA support.
*   **Fast:** Processes millions of tasks per minute with sub-millisecond latency.
*   **Flexible:** Extensible architecture allowing customization of almost every part.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency:** Supports Prefork, Eventlet, gevent, and single-threaded modes.
*   **Result Stores:** Supports various stores like AMQP, Redis, memcached, SQLAlchemy, and others.
*   **Serialization:** Supports pickle, JSON, YAML, msgpack, and compression options.

## What is Celery?

Celery is a task queue, designed to distribute work across threads or machines. It enables asynchronous execution of tasks, allowing your application to remain responsive while handling time-consuming operations in the background.

## Getting Started

Celery runs on:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

### Installation

Install Celery using pip:

```bash
pip install -U Celery
```

### Installation with Bundles

Celery provides a set of bundles to install dependencies for a given feature.

```bash
pip install "celery[redis]"
```

## Core Concepts of Celery

*   **Tasks:** Units of work performed by workers.
*   **Workers:** Processes that execute tasks.
*   **Brokers:** Message brokers like RabbitMQ or Redis that mediate between clients and workers.

## Framework Integration

Celery integrates seamlessly with popular Python web frameworks:

*   **Django:** Built-in integration.
*   **Flask:** Built-in integration.
*   **Pyramid:** Use `pyramid_celery`.
*   **Others:** Support for Pylons, web2py, Tornado, and FastAPI.

## Documentation

*   **Latest Documentation:** [https://docs.celeryq.dev/en/latest/](https://docs.celeryq.dev/en/latest/)

## Contributing

We welcome contributions!  Development happens on GitHub.  See the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) section of the documentation.

## Sponsors

*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

## Support and Community

*   **Mailing List:** Join the `celery-users`_ mailing list.
*   **IRC:** Chat on the #celery channel at Libera Chat.
*   **Bug Tracker:**  Report issues at https://github.com/celery/celery/issues/

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).