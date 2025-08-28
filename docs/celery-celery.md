<div align="center">
  <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner">
  <h1>Celery: Distributed Task Queue for Python</h1>
</div>

<p align="center">
  <a href="https://github.com/celery/celery">
    <img src="https://img.shields.io/github/stars/celery/celery?style=social" alt="GitHub stars">
  </a>
</p>

**Celery is a powerful and easy-to-use distributed task queue that enables asynchronous task processing in Python applications.** 

[View the original repository](https://github.com/celery/celery)

## Key Features

*   **Simple & Flexible:** Easy to use, with no configuration files required. Extend almost any part of Celery to suit your needs.
*   **Highly Available:**  Workers and clients automatically retry in case of connection loss.
*   **Fast:** Can process millions of tasks per minute with low latency.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency:** Includes prefork, Eventlet, gevent, and single-threaded (``solo``) options.
*   **Result Stores:**  AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, Google Cloud Storage, and more.
*   **Serialization:** Supports pickle, json, yaml, msgpack, zlib, bzip2 compression, and cryptographic message signing.
*   **Framework Integration:** Seamless integration with Django, Pyramid, Flask, FastAPI, and more.

## What is a Task Queue?

Task queues distribute work across threads or machines. Clients add *tasks* (units of work) to a queue, and dedicated worker processes constantly monitor and execute those tasks. Celery uses messages, typically with a broker like RabbitMQ or Redis, to manage communication between clients and workers, enabling high availability and horizontal scaling.

## Getting Started

For beginners, explore the "First steps with Celery" and "Next steps" tutorials in the documentation.

### Installation

Install using pip:

```bash
pip install -U Celery
```

Or, install from source:

```bash
# Follow steps to download from PyPI
# Then install by navigating into the directory and running
# python setup.py build
# python setup.py install
```

### Bundles

Celery provides feature-specific bundles:

*   **Serializers:** `auth`, `msgpack`, `yaml`
*   **Concurrency:** `eventlet`, `gevent`
*   **Transports and Backends:** `amqp`, `redis`, `sqs`, `tblib`, `memcache`, `pymemcache`, `cassandra`, `azureblockblob`, `s3`, `gcs`, `couchbase`, `arangodb`, `elasticsearch`, `riak`, `cosmosdbsql`, `zookeeper`, `sqlalchemy`, `pyro`, `slmq`, `consul`, `django`, `gcpubsub`

## Supporting Celery

Celery is supported by:

*   **Open Collective:** [Become a backer](https://opencollective.com/celery#backer)
*   **Sponsors:**  See the list of sponsors below

## Sponsors

### Blacksmith

<a href="https://blacksmith.sh/">
    <img src="./docs/images/blacksmith-logo-white-on-black.svg" alt="Blacksmith logo" width="240">
</a>

### CloudAMQP

<a href="https://www.cloudamqp.com/">
    <img src="./docs/images/cloudamqp-logo-lightbg.svg" alt="CloudAMQP logo" width="240">
</a>

### Upstash

<a href="http://upstash.com/?code=celery">
    <img src="https://upstash.com/logo/upstash-dark-bg.svg" alt="Upstash logo" width="200">
</a>

### Dragonfly

<a href="https://www.dragonflydb.io/">
    <img src="https://github.com/celery/celery/raw/main/docs/images/dragonfly.svg" alt="Dragonfly logo" width="150">
</a>

## Resources

*   **Documentation:** [https://docs.celeryq.dev/en/latest/](https://docs.celeryq.dev/en/latest/)
*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:**  #celery on Libera Chat ([https://libera.chat/](https://libera.chat/))
*   **Bug Tracker:** [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)
*   **Wiki:** [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki)

## Credits

*   **Contributors:** See the [contributors graph](https://github.com/celery/celery/graphs/contributors)
*   **License:** New BSD License.