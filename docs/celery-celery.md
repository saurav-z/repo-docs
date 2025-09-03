# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue that helps you manage asynchronous tasks in your Python applications.** Explore Celery's [GitHub repository](https://github.com/celery/celery) for the latest updates and contributions.

## Key Features

*   **Simple to Use:**  Easy setup and maintenance, with no configuration files needed.
*   **Highly Available:**  Automatic retries and support for high-availability brokers.
*   **Fast:**  Processes millions of tasks per minute with low latency.
*   **Flexible:** Extensible with custom implementations for almost every part.
*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency:** Prefork, Eventlet, gevent, and single-threaded options.
*   **Result Stores:** Supports various options including AMQP, Redis, memcached, SQLAlchemy, and cloud storage solutions like Amazon S3 and Google Cloud Storage.
*   **Serialization:**  Supports pickle, json, yaml, and msgpack, with compression and cryptographic signing options.

## What is a Task Queue?

Task queues enable you to distribute work across threads or machines. Celery uses a message broker to manage tasks, allowing your application to:

1.  A client adds a task to the queue.
2.  The broker delivers it to a worker.

This architecture supports high availability and horizontal scaling with multiple workers and brokers.

## Supported Python Versions

Celery 5.5.x supports:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

For older Python versions, use older Celery versions as specified in the original documentation.

## Get Started

Begin your Celery journey with the following resources:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html) - Learn the bare minimum to get started.
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html) - A more comprehensive overview of features.

## Framework Integration

Celery integrates seamlessly with popular web frameworks:

| Framework       | Integration Package |
| --------------- | ------------------- |
| Django          | Not needed          |
| Pyramid         | pyramid\_celery      |
| Pylons          | celery-pylons       |
| Flask           | Not needed          |
| web2py          | web2py-celery       |
| Tornado         | tornado-celery      |
| FastAPI         | Not needed          |

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

### Bundles
Use bundles to install Celery with specific dependencies for features, e.g., for Redis support:

```bash
pip install "celery[redis]"
```

Available bundles:

*   Serializers (auth, msgpack, yaml)
*   Concurrency (eventlet, gevent)
*   Transports and Backends (amqp, redis, sqs, tblib, memcache, pymemcache, cassandra, azureblockblob, s3, gcs, couchbase, arangodb, elasticsearch, riak, cosmosdbsql, zookeeper, sqlalchemy, pyro, slmq, consul, django, gcpubsub)

## Resources

*   **Documentation:** [Latest Documentation](https://docs.celeryq.dev/en/latest/)
*   **Contributing:**  See the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) guide.
*   **Issue Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)
*   **Wiki:** [GitHub Wiki](https://github.com/celery/celery/wiki)

## Community & Support

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** `#celery` on Libera Chat ([Libera Chat](https://libera.chat/))

## Sponsors

Celery is supported by amazing sponsors! Special thanks to:
*   **Open Collective:**  [Open Collective](https://opencollective.com/celery)
*   **Blacksmith:** [Blacksmith](https://blacksmith.sh/)
*   **CloudAMQP:** [CloudAMQP](https://www.cloudamqp.com/)
*   **Upstash:** [Upstash](http://upstash.com/?code=celery)
*   **Dragonfly:** [Dragonfly](https://www.dragonflydb.io/)

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).  See the `LICENSE` file for details.