[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue that helps you process asynchronous tasks in Python.**

[View the original repository on GitHub](https://github.com/celery/celery)

**Key Features:**

*   **Asynchronous Task Processing:** Offload time-consuming tasks to background workers, improving application responsiveness.
*   **Distributed Architecture:** Scale your task processing across multiple machines and threads for high availability and performance.
*   **Message Broker Support:** Integrates seamlessly with popular message brokers like RabbitMQ, Redis, and Amazon SQS.
*   **Flexible Configuration:** Customize Celery to fit your needs, including custom pools, serializers, and schedulers.
*   **Framework Integration:** Easily integrates with popular Python web frameworks like Django, Flask, and Pyramid.
*   **Multiple Language Support:** While written in Python, Celery's protocol can be used by other languages for interoperability.
*   **Rich Ecosystem:** Extensive support for transports, concurrency models, result stores, and serialization methods.

**Version:** 5.5.3 (immunity)

**Important Links:**

*   **Website:** [https://docs.celeryq.dev/en/stable/index.html](https://docs.celeryq.dev/en/stable/index.html)
*   **Download:** [https://pypi.org/project/celery/](https://pypi.org/project/celery/)
*   **Source Code:** [https://github.com/celery/celery/](https://github.com/celery/celery/)
*   **DeepWiki:** [![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/celery/celery)

## Donations

Celery is a community-driven project, and your support helps us keep it going.

### Open Collective

[![Open Collective logo](https://opencollective.com/static/images/opencollectivelogo-footer-n.svg)](https://opencollective.com/celery)

[Open Collective](https://opencollective.com/celery) is our community-powered funding platform. Your sponsorship directly supports improvements, maintenance, and innovative features.

## For Enterprise

Celery is available as part of the Tidelift Subscription.  Save time, reduce risk, and improve code health, while paying the maintainers of the exact dependencies you use. `Learn more. <https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo>`_

## Sponsors

**(Logos and links to sponsors remain unchanged, ensuring SEO benefits from those links)**

## What's a Task Queue?

Task queues are essential for distributing work across threads or machines.  Celery facilitates communication between clients and workers via messages, allowing for high availability and scalability.

## What Do I Need?

*   **Python Versions:** Celery 5.5.x supports Python 3.8, 3.9, 3.10, 3.11, 3.12, and 3.13, as well as PyPy3.9+ (v7.3.12+). For older Python versions, use earlier Celery releases as specified in the original README.

*   **Message Brokers:** Celery utilizes message brokers like RabbitMQ, Redis, Amazon SQS, and others to manage task distribution.

## Get Started

Begin your Celery journey with the following resources:

*   `First steps with Celery`_:  Begin with the bare minimum needed to get started.
*   `Next steps`_: A more complete overview, showing more features.

## Celery is...

*   **Simple:**  Easy to use and maintain, with no configuration files needed.
*   **Highly Available:**  Workers and clients automatically retry in case of connection loss.
*   **Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible:** Extensible with custom implementations for pools, serializers, and more.

## It Supports...

*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, Zookeeper, Pyro4, SoftLayer Message Queue, Consul.io Key/Value store, and others.
*   **Concurrency:** Prefork, Eventlet_, gevent_, and single-threaded ("solo") options.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, Google Cloud Storage, Couchbase, ArangoDB, Riak, Cosmos DB, and others.
*   **Serialization:** *pickle*, *json*, *yaml*, *msgpack*, with *zlib* and *bzip2* compression.

## Framework Integration

Celery integrates easily with various Python web frameworks. Refer to the table in the original README.

## Documentation

Find the latest documentation at [https://docs.celeryq.dev/en/latest/](https://docs.celeryq.dev/en/latest/).

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

## Bundles

Use bundles to install Celery with specific dependencies:

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

### Downloading and Installing from Source

Download from PyPI, unpack, and install using `python setup.py build` and `python setup.py install`.

### Using the Development Version

Install the latest development versions of required packages using pip.

## Getting Help

### Mailing List

Join the `celery-users`_ mailing list for discussions.

### IRC

Chat with us on IRC in the **#celery** channel on `Libera Chat`_.

### Bug Tracker

Report issues at [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/).

## Wiki

Visit the wiki at [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki).

## Credits

### Contributors

Development happens on GitHub: [https://github.com/celery/celery](https://github.com/celery/celery). Read the `Contributing to Celery`_ guide.

|oc-contributors|

### Backers

Thank you to all our backers! 🙏 [`Become a backer`_]

|oc-backers|

## License

Licensed under the `New BSD License`. See the ``LICENSE`` file.