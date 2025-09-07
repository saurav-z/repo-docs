[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful, easy-to-use distributed task queue that allows you to run asynchronous tasks in the background.**

[Link to Original Repo](https://github.com/celery/celery)

Key Features:

*   **Simple to Use:** Celery boasts a straightforward setup and configuration, eliminating the need for extensive configuration files.
*   **Highly Available:** Ensures task reliability with automatic retries for connection issues and broker support for high availability.
*   **Fast Performance:** Capable of processing millions of tasks per minute with low latency.
*   **Flexible and Extensible:**  Offers numerous customization options for custom pool implementations, serializers, transports, and more.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more
*   **Concurrency:**  Offers Prefork, Eventlet, gevent, and single-threaded (``solo``) concurrency models.
*   **Result Stores:**  Provides support for various result stores including AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, and Google Cloud Storage.
*   **Serialization:** Supports *pickle*, *json*, *yaml*, *msgpack*, and *zlib*, *bzip2* compression, with cryptographic message signing.

## Core Concepts: What is a Task Queue?

Task queues distribute work across threads or machines, enabling asynchronous task execution.  Celery utilizes a message broker to facilitate communication between clients and workers. A client enqueues a task, the broker delivers it to a worker for processing.  This architecture allows for high availability and scalability.

## Getting Started with Celery

Follow these resources for a quick start:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

### Requirements

Celery version 5.5.x supports:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

Older Celery versions support older Python versions (see the original README for details).

## Framework Integration

Celery seamlessly integrates with popular web frameworks. Here's a summary:

| Framework      | Integration Package      |
| :------------- | :----------------------- |
| Django         | Not Needed               |
| Pyramid        | `pyramid_celery`         |
| Pylons         | `celery-pylons`          |
| Flask          | Not Needed               |
| web2py         | `web2py-celery`          |
| Tornado        | `tornado-celery`         |
| FastAPI        | Not Needed               |

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

### Bundles

Celery supports installation with specific bundles for features:

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

## Documentation and Resources

*   [Latest Documentation](https://docs.celeryq.dev/en/latest/) - User guides, tutorials, and API reference.
*   [Celery Wiki](https://github.com/celery/celery/wiki)

## Getting Help

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** `#celery` on Libera Chat ([Libera Chat](https://libera.chat/))
*   **Bug Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)

## Sponsors

Celery is supported by generous sponsors:

*   [Open Collective](https://opencollective.com/celery)
*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](https://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

## Contributing

Contribute to Celery by reading the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) and development takes place on GitHub.

## Credits

*   [Contributors](https://github.com/celery/celery/graphs/contributors) - Thanks to all contributors!
*   [Backers](https://opencollective.com/celery#backers) - Thank you to all our backers!

## License

Licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).