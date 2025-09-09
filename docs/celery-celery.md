# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue, perfect for asynchronous processing and background tasks in your Python applications.**  [See the original repo](https://github.com/celery/celery).

[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep Security](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementations](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)
[![Open Collective Backers](https://opencollective.com/celery/backers/badge.svg)](#backers)
[![Open Collective Sponsors](https://opencollective.com/celery/sponsors/badge.svg)](#sponsors)

*Version: 5.5.3 (immunity)*

## Key Features

*   **Simple:** Easy to learn and use, with minimal configuration.
*   **Highly Available:** Automatic retries and support for HA brokers.
*   **Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible:** Extensible with custom pool implementations, serializers, and more.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency:** Prefork, Eventlet, gevent, and single-threaded options.
*   **Result Stores:**  AMQP, Redis, memcached, SQLAlchemy, and cloud-based options.

## Donations

### Open Collective

<a href="https://opencollective.com/celery" target="_blank"><img src="https://opencollective.com/static/images/opencollectivelogo-footer-n.svg" alt="Open Collective logo" width="200"></a>

Support Celery's development through Open Collective.  Your sponsorship helps maintain and improve Celery.

## For Enterprise

[Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo) offers commercial support and maintenance for Celery and thousands of other packages.

## Sponsors

**Blacksmith**

<a href="https://blacksmith.sh/" target="_blank"><img src="./docs/images/blacksmith-logo-white-on-black.svg" alt="Blacksmith logo" width="240"></a>

**CloudAMQP**

<a href="https://www.cloudamqp.com/" target="_blank"><img src="./docs/images/cloudamqp-logo-lightbg.svg" alt="CloudAMQP logo" width="240"></a>

**Upstash**

<a href="http://upstash.com/?code=celery" target="_blank"><img src="https://upstash.com/logo/upstash-dark-bg.svg" alt="Upstash logo" width="200"></a>

**Dragonfly**

<a href="https://www.dragonflydb.io/" target="_blank"><img src="https://github.com/celery/celery/raw/main/docs/images/dragonfly.svg" alt="Dragonfly logo" width="150"></a>

## What's a Task Queue?

Task queues distribute work across threads or machines. Celery utilizes a message broker to manage tasks and workers.

## What do I need?

Celery 5.5.x runs on:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

Older Python versions require older Celery versions.

## Get Started

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

CloudAMQP, a leading RabbitMQ hosting provider, also provides a great starting point.

## Celery is...

*   **Simple** - Easy to use and maintain with an active community.
*   **Highly Available** - Retries and broker HA support.
*   **Fast** - Processes millions of tasks with low latency.
*   **Flexible** - Extensible for custom needs.

## It supports...

*   **Message Transports** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub
*   **Concurrency** Prefork, Eventlet, gevent, single threaded (``solo``)
*   **Result Stores** AMQP, Redis, memcached, SQLAlchemy, Django ORM, cloud options.
*   **Serialization** pickle, json, yaml, msgpack; zlib, bzip2 compression; cryptographic signing.

## Framework Integration

Celery integrates easily with popular Python web frameworks:

| Framework    | Integration |
|--------------|-------------|
| Django       | not needed  |
| Pyramid      | pyramid\_celery |
| Pylons       | celery-pylons |
| Flask        | not needed  |
| web2py       | web2py-celery |
| Tornado      | tornado-celery |
| FastAPI      | not needed  |

## Documentation

*   [Latest Documentation](https://docs.celeryq.dev/en/latest/)

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

### Bundles

Install Celery with specific features using bundles:

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

Available bundles:  (serializers, concurrency, transports/backends)

### Downloading and installing from source

Download from PyPI and install:

```bash
$ tar xvfz celery-0.0.0.tar.gz
$ cd celery-0.0.0
$ python setup.py build
$ # python setup.py install
```

### Using the development version

Install development dependencies using pip:

```bash
$ pip install https://github.com/celery/celery/zipball/main#egg=celery
$ pip install https://github.com/celery/billiard/zipball/main#egg=billiard
$ pip install https://github.com/celery/py-amqp/zipball/main#egg=amqp
$ pip install https://github.com/celery/kombu/zipball/main#egg=kombu
$ pip install https://github.com/celery/vine/zipball/main#egg=vine
```

## Getting Help

### Mailing list

*   [celery-users mailing list](https://groups.google.com/group/celery-users/)

### IRC

*   Join the **#celery** channel on [Libera Chat](https://libera.chat/)

## Bug tracker

*   [Issue tracker](https://github.com/celery/celery/issues/)

## Wiki

*   [Wiki](https://github.com/celery/celery/wiki)

## Credits

### Contributors

Development happens at GitHub:  [https://github.com/celery/celery](https://github.com/celery/celery)

### Backers

Thank you to all our backers! 🙏  [Become a backer](https://opencollective.com/celery#backer)

|oc-backers|

### License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).  See the LICENSE file.