<div align="center">
  <a href="https://github.com/celery/celery">
    <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner" width="80%">
  </a>
</div>

# Celery: The Distributed Task Queue for Python

**Celery is an easy-to-use, highly available, and fast task queue that empowers you to distribute work across threads or machines.**

[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage Status](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep Security](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementations](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)
[![Open Collective Backers](https://opencollective.com/celery/backers/badge.svg)](https://opencollective.com/celery#backers)
[![Open Collective Sponsors](https://opencollective.com/celery/sponsors/badge.svg)](https://opencollective.com/celery#sponsors)
[![Downloads](https://pepy.tech/badge/celery)](https://pepy.tech/project/celery)
<a href="https://deepwiki.com/celery/celery"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki" width="125px"></a>
<br/>
[Celery Documentation](https://docs.celeryq.dev/en/stable/index.html) | [PyPI](https://pypi.org/project/celery/) | [Source Code](https://github.com/celery/celery)

## Key Features

*   **Simple:** Easy to use and maintain, with minimal configuration.
*   **Highly Available:** Workers and clients automatically retry in case of connection loss.
*   **Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible:** Extensible with custom pool implementations, serializers, and more.
*   **Language Agnostic:** While written in Python, the protocol can be implemented in any language.

## What is a Task Queue?

Task queues are used to distribute work across threads or machines. A task queue's input is a unit of work, called a task. Dedicated worker processes then constantly monitor the queue for new work to perform.

Celery communicates via messages, usually using a broker to mediate between clients and workers. To initiate a task a client puts a message on the queue, the broker then delivers the message to a worker.

## Key Benefits

*   **Asynchronous Task Execution:** Run tasks in the background, freeing up your application's resources.
*   **Distributed Processing:** Scale your application by distributing tasks across multiple workers and machines.
*   **Reliability:** Built-in mechanisms for retrying failed tasks and handling connection issues.
*   **Scalability:** Easily handle increasing workloads by adding more workers.

## Celery Ecosystem

Celery supports several message brokers, concurrency frameworks, result stores, and serialization options.

### Message Transports

*   RabbitMQ
*   Redis
*   Amazon SQS
*   Google Pub/Sub

### Concurrency

*   Prefork
*   Eventlet
*   Gevent
*   Single threaded (``solo``)

### Result Stores

*   AMQP
*   Redis
*   memcached
*   SQLAlchemy, Django ORM
*   Apache Cassandra, IronCache, Elasticsearch
*   Google Cloud Storage

### Serialization

*   pickle, json, yaml, msgpack
*   zlib, bzip2 compression
*   Cryptographic message signing

## Framework Integration

Celery integrates seamlessly with popular web frameworks, with many having dedicated integration packages:

| Framework      | Integration                               |
| --------------- | ----------------------------------------- |
| Django          | not needed                                |
| Pyramid         | `pyramid_celery`                          |
| Pylons          | `celery-pylons`                           |
| Flask           | not needed                                |
| web2py          | `web2py-celery`                           |
| Tornado         | `tornado-celery`                          |
| FastAPI         | not needed                                |

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

Or, use specific bundles:

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

See [Installation](#installation) in the documentation for more details.

## Getting Started

Explore the Celery documentation:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Sponsors

Celery is supported by awesome sponsors, including:

*   **Blacksmith**
    <br/>
    <a href="https://blacksmith.sh/" target="_blank"><img src="./docs/images/blacksmith-logo-white-on-black.svg" alt="Blacksmith logo" width="240px"/></a>
*   **CloudAMQP**
    <br/>
    <a href="https://www.cloudamqp.com/" target="_blank"><img src="./docs/images/cloudamqp-logo-lightbg.svg" alt="CloudAMQP logo" width="240px"/></a>
*   **Upstash**
    <br/>
    <a href="http://upstash.com/?code=celery" target="_blank"><img src="https://upstash.com/logo/upstash-dark-bg.svg" alt="Upstash logo" width="200px"/></a>
*   **Dragonfly**
    <br/>
    <a href="https://www.dragonflydb.io/" target="_blank"><img src="https://github.com/celery/celery/raw/main/docs/images/dragonfly.svg" alt="Dragonfly logo" width="150px"/></a>

## Contributing

We welcome contributions! Read the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) guide. The Celery development happens at GitHub: [https://github.com/celery/celery](https://github.com/celery/celery)

## Support

*   [Mailing list](https://groups.google.com/group/celery-users/)
*   [IRC](https://libera.chat/) -  **#celery** on Libera Chat
*   [Bug Tracker](https://github.com/celery/celery/issues/)
*   [Wiki](https://github.com/celery/celery/wiki)

## License

This software is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause). See the ``LICENSE`` file for details.

**[Back to Top](#celery-the-distributed-task-queue-for-python)**