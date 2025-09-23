[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and flexible distributed task queue that enables asynchronous task execution.**

[Visit the Celery Repository on GitHub](https://github.com/celery/celery)

## Key Features

*   **Simple:** Easy to use and maintain, minimizing configuration.
*   **Highly Available:**  Automatic retries and support for High Availability in brokers.
*   **Fast:** Processes millions of tasks per minute with sub-millisecond latency.
*   **Flexible:** Extensible with custom pools, serializers, transports, and more.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and others.
*   **Concurrency:** Includes Prefork, Eventlet, gevent, and single-threaded (solo) options.
*   **Result Stores:**  Offers various options like AMQP, Redis, memcached, SQLAlchemy, and cloud storage services.
*   **Serialization:**  Supports pickle, JSON, YAML, msgpack with compression and cryptographic message signing.

## What is a Task Queue?

Task queues facilitate the distribution of work across threads or machines. Celery utilizes message brokers to manage task distribution: clients send messages containing tasks, brokers deliver them to workers, and workers execute these tasks. This architecture enables high availability and horizontal scaling.

## Getting Started

To learn how to use Celery, check out these getting started tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Framework Integration

Celery seamlessly integrates with popular Python web frameworks.

*   **Django:** No additional packages are needed.
*   **Pyramid:** [pyramid\_celery](https://pypi.org/project/pyramid_celery/)
*   **Pylons:** [celery-pylons](https://pypi.org/project/celery-pylons/)
*   **Flask:** No additional packages are needed.
*   **web2py:** [web2py-celery](https://code.google.com/p/web2py-celery/)
*   **Tornado:** [tornado-celery](https://github.com/mher/tornado-celery/)
*   **FastAPI:** No additional packages are needed.

## Installation

Install Celery using `pip`:

```bash
pip install -U Celery
```

Celery offers bundles for simplified installation of specific dependencies:

*   `[auth]`: Security serializer.
*   `[msgpack]`: msgpack serializer.
*   `[yaml]`: YAML serializer.
*   `[eventlet]`: eventlet pool.
*   `[gevent]`: gevent pool.
*   `[amqp]`: RabbitMQ amqp library.
*   `[redis]`: Redis transport or result backend.
*   `[sqs]`: Amazon SQS transport.
*   `[tblib]`: task_remote_tracebacks feature.
*   `[memcache]`: Memcached result backend (pylibmc).
*   `[pymemcache]`: Memcached result backend (pure-Python).
*   `[cassandra]`: Cassandra/Astra DB result backend (DataStax driver).
*   `[azureblockblob]`: Azure Storage result backend.
*   `[s3]`: S3 Storage result backend.
*   `[gcs]`: Google Cloud Storage result backend.
*   `[couchbase]`: Couchbase result backend.
*   `[arangodb]`: ArangoDB result backend.
*   `[elasticsearch]`: Elasticsearch result backend.
*   `[riak]`: Riak result backend.
*   `[cosmosdbsql]`: Azure Cosmos DB result backend (pydocumentdb).
*   `[zookeeper]`: Zookeeper transport.
*   `[sqlalchemy]`: SQLAlchemy result backend.
*   `[pyro]`: Pyro4 transport (*experimental*).
*   `[slmq]`: SoftLayer Message Queue transport (*experimental*).
*   `[consul]`: Consul.io Key/Value store as a message transport or result backend (*experimental*).
*   `[django]`: Django support.
*   `[gcpubsub]`: Google Pub/Sub transport.

## Documentation

*   [Latest Documentation](https://docs.celeryq.dev/en/latest/)

## Support & Community

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:**  #celery on Libera Chat ([Libera Chat](https://libera.chat/))
*   **Bug Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)

## Sponsors and Backers

Celery's development is supported by generous sponsors and backers. [Become a backer](https://opencollective.com/celery#backer).

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).