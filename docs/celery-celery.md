<!-- Banner Image - SEO Friendly -->
<p align="center">
  <a href="https://github.com/celery/celery">
    <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner" width="100%">
  </a>
</p>

<!-- Badges - Keep for quick info -->
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
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/celery/celery)

# Celery: Distributed Task Queue for Python - Asynchronous Task Processing Made Easy

Celery is a robust, distributed task queue that helps you handle asynchronous tasks in Python with ease, enabling scalability and efficiency. ([Original Repo](https://github.com/celery/celery))

**Key Features:**

*   **Simplicity:** Easy to use and maintain, requiring no configuration files.
*   **High Availability:** Automatic retries for robust task execution.
*   **Speed:** Processes millions of tasks per minute.
*   **Flexibility:** Extensible and customizable, with support for many options.

**Core Functionality:**

*   **Task Distribution:** Distributes work across threads or machines.
*   **Message-Driven:** Uses a message broker for communication between clients and workers.
*   **Scalability:** Supports multiple workers and brokers for high availability and horizontal scaling.
*   **Language Interoperability:** Protocol can be implemented in any language.

**Supported Features:**

*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub.
*   **Concurrency:** Prefork, Eventlet, gevent, single threaded (``solo``).
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, Google Cloud Storage.
*   **Serialization:** pickle, json, yaml, msgpack with zlib, bzip2 compression and Cryptographic message signing.

## Getting Started

To quickly get started with Celery, refer to the following tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Framework Integration

Celery seamlessly integrates with popular web frameworks:

*   Django (no integration package needed)
*   Pyramid (pyramid_celery)
*   Pylons (celery-pylons)
*   Flask (no integration package needed)
*   web2py (web2py-celery)
*   Tornado (tornado-celery)
*   FastAPI (no integration package needed)

## Installation

Install Celery using `pip`:

```bash
pip install -U Celery
```

You can also install optional bundles for specific features:

*   `celery[auth]`
*   `celery[msgpack]`
*   `celery[yaml]`
*   `celery[eventlet]`
*   `celery[gevent]`
*   `celery[amqp]`
*   `celery[redis]`
*   `celery[sqs]`
*   `celery[tblib]`
*   `celery[memcache]`
*   `celery[pymemcache]`
*   `celery[cassandra]`
*   `celery[azureblockblob]`
*   `celery[s3]`
*   `celery[gcs]`
*   `celery[couchbase]`
*   `celery[arangodb]`
*   `celery[elasticsearch]`
*   `celery[riak]`
*   `celery[cosmosdbsql]`
*   `celery[zookeeper]`
*   `celery[sqlalchemy]`
*   `celery[pyro]`
*   `celery[slmq]`
*   `celery[consul]`
*   `celery[django]`
*   `celery[gcpubsub]`

### Downloading and installing from source

```bash
$ pip install -U Celery
```

```bash
$ tar xvfz celery-0.0.0.tar.gz
$ cd celery-0.0.0
$ python setup.py build
# python setup.py install
```

## Community and Support

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** #celery on Libera Chat
*   **Issue Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)
*   **Wiki:** [GitHub Wiki](https://github.com/celery/celery/wiki)

## Contributing

We welcome contributions!  Development happens at GitHub: https://github.com/celery/celery. Read the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) section in the documentation.

## Sponsors

Celery is supported by the community.

*   **Open Collective:** [Become a backer](https://opencollective.com/celery#backer)
    [![Open Collective Backers](https://opencollective.com/celery/backers.svg?width=890)](https://opencollective.com/celery#backers)

*   **Sponsors:**

    *   Blacksmith: [Blacksmith](https://blacksmith.sh/)
    *   CloudAMQP: [CloudAMQP](https://www.cloudamqp.com/)
    *   Upstash: [Upstash](http://upstash.com/?code=celery)
    *   Dragonfly: [Dragonfly](https://www.dragonflydb.io/)

## License

This project is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).