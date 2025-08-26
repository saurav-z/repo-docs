<div align="center">
  <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner" width="800">
</div>

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue that helps you execute asynchronous tasks, manage background jobs, and build scalable applications.**  [See the original repository](https://github.com/celery/celery).

[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage Status](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep Security](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementation](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)
[![Open Collective Backers](https://opencollective.com/celery/backers/badge.svg)](https://opencollective.com/celery#backers)
[![Open Collective Sponsors](https://opencollective.com/celery/sponsors/badge.svg)](https://opencollective.com/celery#sponsors)

**Version:** 5.5.3 (immunity)

**Key Features:**

*   **Simplicity:** Easy to use, maintain, and configure.
*   **High Availability:** Automatic retries and support for primary/primary or primary/replica replication with some brokers.
*   **Speed:** Capable of processing millions of tasks per minute with low latency.
*   **Flexibility:** Extensible for custom pools, serializers, logging, and more.
*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and others.
*   **Concurrency:** Prefork, Eventlet, gevent, and single-threaded options.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, and more.
*   **Serialization:** Pickle, JSON, YAML, msgpack, and compression options.

## What is a Task Queue?

Task queues are used to distribute work across threads or machines. Celery facilitates this by:

*   Breaking down work into tasks.
*   Worker processes monitoring a queue for new tasks.
*   Clients putting messages on the queue, and brokers delivering them to workers.
*   Supporting multiple workers and brokers for high availability and scaling.

## Getting Started

**Prerequisites:**

*   Python 3.8+
*   PyPy3.9+ (v7.3.12+)

Refer to the [Getting Started Tutorials](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html) for detailed instructions.

## Framework Integration

Celery seamlessly integrates with popular web frameworks:

| Framework       | Integration Package     |
| --------------- | ----------------------- |
| Django          | N/A                     |
| Pyramid         | pyramid_celery          |
| Pylons          | celery-pylons           |
| Flask           | N/A                     |
| web2py          | web2py-celery           |
| Tornado         | tornado-celery          |
| FastAPI         | N/A                     |

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

For specific features, you can install bundles like:

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

See the original README for a full list of bundles.

### Installing from Source

Download and install from source:

```bash
$ tar xvfz celery-0.0.0.tar.gz
$ cd celery-0.0.0
$ python setup.py build
# python setup.py install
```

## Community & Support

*   **Mailing List:** Join the `celery-users` mailing list for discussions.
*   **IRC:** Chat with us on IRC in the `#celery` channel at Libera Chat.
*   **Bug Tracker:** Report issues and suggestions at the [issue tracker](https://github.com/celery/celery/issues/).
*   **Wiki:** Explore the [wiki](https://github.com/celery/celery/wiki) for additional information.

## Sponsors & Support

Celery is supported by generous sponsors:

*   [Open Collective](https://opencollective.com/celery) - Community-powered funding.
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo) - Commercial Support and Maintenance
*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

## Contributing

Contributions are welcome!  Read the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) section in the documentation.

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).