<div align="center">
  <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner" />
</div>

# Celery: Distributed Task Queue for Python

**Celery is a powerful and flexible distributed task queue that enables you to process tasks asynchronously, making your applications more responsive and scalable.** [See the original repository](https://github.com/celery/celery)

[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage Status](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementation](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)
[![Open Collective Backers](https://opencollective.com/celery/backers/badge.svg)](https://opencollective.com/celery#backers)
[![Open Collective Sponsors](https://opencollective.com/celery/sponsors/badge.svg)](https://opencollective.com/celery#sponsors)
[![Downloads](https://pepy.tech/badge/celery)](https://pepy.tech/project/celery)
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/celery/celery)

**Key Features:**

*   **Simple:** Easy to use and maintain with no configuration files needed.
*   **Highly Available:** Workers and clients automatically retry in case of connection loss.
*   **Fast:** Capable of processing millions of tasks per minute with low latency.
*   **Flexible:** Extensible, supporting custom pools, serializers, compression, and more.
*   **Wide Broker Support:**  RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and experimental support for others.
*   **Concurrency:** Supports Prefork, Eventlet, gevent, and single-threaded (solo) models.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, Django ORM, and more.
*   **Serialization:**  pickle, json, yaml, msgpack with zlib, bzip2 compression, and cryptographic message signing.
*   **Framework Integration:** Seamlessly integrates with Django, Flask, Pyramid, and more.

## Table of Contents

*   [What's a Task Queue?](#whats-a-task-queue)
*   [What do I need?](#what-do-i-need)
*   [Get Started](#get-started)
*   [Celery is...](#celery-is)
*   [It supports...](#it-supports)
*   [Framework Integration](#framework-integration)
*   [Documentation](#documentation)
*   [Installation](#installation)
    *   [Installation via pip](#installation-via-pip)
    *   [Bundles](#bundles)
    *   [Downloading and installing from source](#downloading-and-installing-from-source)
    *   [Using the development version](#using-the-development-version)
        *   [With pip](#with-pip)
        *   [With git](#with-git)
*   [Getting Help](#getting-help)
    *   [Mailing list](#mailing-list)
    *   [IRC](#irc)
*   [Bug tracker](#bug-tracker)
*   [Wiki](#wiki)
*   [Credits](#credits)
    *   [Contributors](#contributors)
    *   [Backers](#backers)
*   [License](#license)
*   [Sponsors](#sponsors)

## What's a Task Queue?

Task queues are used to distribute work across threads or machines. Celery utilizes task queues to enable asynchronous task processing.

## What do I need?

Celery v5.5.x runs on:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

If you're running an older version of Python, you need to be running an older version of Celery.

Celery is usually used with a message broker. The RabbitMQ, Redis transports are feature complete, but there's also experimental support for a myriad of other solutions, including using SQLite for local development.

Celery can run on a single machine, on multiple machines, or even across datacenters.

## Get Started

If this is the first time you're trying to use Celery, or you're new to Celery v5.5.x coming from previous versions then you should read our getting started tutorials:

*   [`First steps with Celery`](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
    *   Tutorial teaching you the bare minimum needed to get started with Celery.

*   [`Next steps`](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)
    *   A more complete overview, showing more features.

 You can also get started with Celery by using a hosted broker transport CloudAMQP. The largest hosting provider of RabbitMQ is a proud sponsor of Celery.

## Celery is...

*   **Simple**

    Celery is easy to use and maintain, and does *not need configuration files*.

    It has an active, friendly community you can talk to for support,
    like at our `mailing-list`_, or the IRC channel.

    Here's one of the simplest applications you can make:

    ```python
    from celery import Celery

    app = Celery('hello', broker='amqp://guest@localhost//')

    @app.task
    def hello():
        return 'hello world'
    ```

*   **Highly Available**

    Workers and clients will automatically retry in the event
    of connection loss or failure, and some brokers support
    HA in way of *Primary/Primary* or *Primary/Replica* replication.

*   **Fast**

    A single Celery process can process millions of tasks a minute,
    with sub-millisecond round-trip latency (using RabbitMQ,
    py-librabbitmq, and optimized settings).

*   **Flexible**

    Almost every part of *Celery* can be extended or used on its own,
    Custom pool implementations, serializers, compression schemes, logging,
    schedulers, consumers, producers, broker transports, and much more.

## It supports...

*   **Message Transports**

    *   RabbitMQ, Redis, Amazon SQS, Google Pub/Sub

*   **Concurrency**

    *   Prefork, Eventlet, gevent, single threaded (``solo``)

*   **Result Stores**

    *   AMQP, Redis
    *   memcached
    *   SQLAlchemy, Django ORM
    *   Apache Cassandra, IronCache, Elasticsearch
    *   Google Cloud Storage

*   **Serialization**

    *   *pickle*, *json*, *yaml*, *msgpack*.
    *   *zlib*, *bzip2* compression.
    *   Cryptographic message signing.

## Framework Integration

Celery easily integrates with web frameworks.

| Framework | Integration |
| ------------- | ------------- |
| Django | not needed |
| Pyramid |  `pyramid_celery` |
| Pylons |  `celery-pylons`  |
| Flask  | not needed  |
| web2py | `web2py-celery`  |
| Tornado  | `tornado-celery`  |
| FastAPI  | not needed  |

## Documentation

The [`latest documentation`](https://docs.celeryq.dev/en/latest/) is hosted at Read The Docs.

## Installation

### Installation via pip

```bash
pip install -U Celery
```

### Bundles

Celery also defines a group of bundles that can be used to install Celery and the dependencies for a given feature.

You can specify these in your requirements or on the ``pip`` command-line by using brackets. Multiple bundles can be specified by separating them by commas.

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

The following bundles are available:

**Serializers**

*   ``celery[auth]``: for using the ``auth`` security serializer.
*   ``celery[msgpack]``: for using the msgpack serializer.
*   ``celery[yaml]``: for using the yaml serializer.

**Concurrency**

*   ``celery[eventlet]``: for using the ``eventlet`` pool.
*   ``celery[gevent]``: for using the ``gevent`` pool.

**Transports and Backends**

*   ``celery[amqp]``: for using the RabbitMQ amqp python library.
*   ``celery[redis]``: for using Redis as a message transport or as a result backend.
*   ``celery[sqs]``: for using Amazon SQS as a message transport.
*   ``celery[tblib``]: for using the ``task_remote_tracebacks`` feature.
*   ``celery[memcache]``: for using Memcached as a result backend (using ``pylibmc``)
*   ``celery[pymemcache]``: for using Memcached as a result backend (pure-Python implementation).
*   ``celery[cassandra]``: for using Apache Cassandra/Astra DB as a result backend with the DataStax driver.
*   ``celery[azureblockblob]``: for using Azure Storage as a result backend (using ``azure-storage``)
*   ``celery[s3]``: for using S3 Storage as a result backend.
*   ``celery[gcs]``: for using Google Cloud Storage as a result backend.
*   ``celery[couchbase]``: for using Couchbase as a result backend.
*   ``celery[arangodb]``: for using ArangoDB as a result backend.
*   ``celery[elasticsearch]``: for using Elasticsearch as a result backend.
*   ``celery[riak]``: for using Riak as a result backend.
*   ``celery[cosmosdbsql]``: for using Azure Cosmos DB as a result backend (using ``pydocumentdb``)
*   ``celery[zookeeper]``: for using Zookeeper as a message transport.
*   ``celery[sqlalchemy]``: for using SQLAlchemy as a result backend (*supported*).
*   ``celery[pyro]``: for using the Pyro4 message transport (*experimental*).
*   ``celery[slmq]``: for using the SoftLayer Message Queue transport (*experimental*).
*   ``celery[consul]``: for using the Consul.io Key/Value store as a message transport or result backend (*experimental*).
*   ``celery[django]``: specifies the lowest version possible for Django support.
    *   You should probably not use this in your requirements, it's here for informational purposes only.
*   ``celery[gcpubsub]``: for using Google Pub/Sub as a message transport.

### Downloading and installing from source

Download the latest version of Celery from PyPI:
https://pypi.org/project/celery/

```bash
tar xvfz celery-0.0.0.tar.gz
cd celery-0.0.0
python setup.py build
# python setup.py install
```

### Using the development version

#### With pip

```bash
pip install https://github.com/celery/celery/zipball/main#egg=celery
pip install https://github.com/celery/billiard/zipball/main#egg=billiard
pip install https://github.com/celery/py-amqp/zipball/main#egg=amqp
pip install https://github.com/celery/kombu/zipball/main#egg=kombu
pip install https://github.com/celery/vine/zipball/main#egg=vine
```

#### With git

Please see the Contributing section.

## Getting Help

### Mailing list

For discussions about the usage, development, and future of Celery, please join the [`celery-users` mailing list](https://groups.google.com/group/celery-users/).

### IRC

Come chat with us on IRC. The **#celery** channel is located at the [`Libera Chat`](https://libera.chat/) network.

## Bug tracker

If you have any suggestions, bug reports, or annoyances please report them to our issue tracker at https://github.com/celery/celery/issues/

## Wiki

https://github.com/celery/celery/wiki

## Credits

### Contributors

This project exists thanks to all the people who contribute. Development of `celery` happens at GitHub: https://github.com/celery/celery

You're highly encouraged to participate in the development of `celery`. If you don't like GitHub (for some reason) you're welcome to send regular patches.

Be sure to also read the [`Contributing to Celery`](https://docs.celeryq.dev/en/stable/contributing.html) section in the documentation.

|oc-contributors|

### Backers

Thank you to all our backers! 🙏 [`Become a backer`](https://opencollective.com/celery#backer)

|oc-backers|

## License

This software is licensed under the [`New BSD License`](https://opensource.org/licenses/BSD-3-Clause). See the ``LICENSE`` file in the top distribution directory for the full license text.

## Sponsors

### Blacksmith

<div align="center">
  <a href="https://blacksmith.sh/" target="_blank">
    <img src="./docs/images/blacksmith-logo-white-on-black.svg" alt="Blacksmith logo" width="240px"/>
  </a>
</div>

### CloudAMQP

<div align="center">
  <a href="https://www.cloudamqp.com/" target="_blank">
    <img src="./docs/images/cloudamqp-logo-lightbg.svg" alt="CloudAMQP logo" width="240px"/>
  </a>
</div>

### Upstash

<div align="center">
  <a href="http://upstash.com/?code=celery" target="_blank">
    <img src="https://upstash.com/logo/upstash-dark-bg.svg" alt="Upstash logo" width="200px"/>
  </a>
</div>

### Dragonfly

<div align="center">
  <a href="https://www.dragonflydb.io/" target="_blank">
    <img src="https://github.com/celery/celery/raw/main/docs/images/dragonfly.svg" alt="Dragonfly logo" width="150px"/>
  </a>
</div>