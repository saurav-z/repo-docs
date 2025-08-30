<div align="center">
  <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner" width="800"/>
  <br>
  <a href="https://github.com/celery/celery">
    <img src="https://img.shields.io/github/stars/celery/celery?style=social" alt="Stars"/>
  </a>
</div>

# Celery: Distributed Task Queue for Python

**Celery is a powerful and flexible distributed task queue that enables you to process asynchronous tasks in Python.**

**Key Features:**

*   **Simple to Use:** Easy setup and maintenance, no configuration files needed.
*   **Highly Available:** Automatic retries and support for high availability brokers.
*   **Fast Performance:** Processes millions of tasks per minute with low latency.
*   **Flexible & Extensible:** Customize almost every aspect of Celery.
*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency:** Prefork, Eventlet, gevent, and single-threaded options.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, and various other options.
*   **Serialization:** pickle, json, yaml, msgpack with compression and cryptographic signing.

## Core Concepts

Celery uses a message broker to facilitate communication between clients (who enqueue tasks) and workers (who execute tasks). This architecture allows for:

*   **Asynchronous Task Execution:** Offload time-consuming operations to the background.
*   **Distributed Processing:** Scale your application by running workers on multiple machines.
*   **Decoupling:** Separate your application's components for better maintainability.

## Getting Started

1.  **Installation:**
    ```bash
    pip install -U Celery
    ```

2.  **Basic Example:**

    ```python
    from celery import Celery

    app = Celery('hello', broker='amqp://guest@localhost//')

    @app.task
    def hello():
        return 'hello world'
    ```

    This example defines a simple Celery task that can be executed asynchronously.  See the [Getting Started Tutorials](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html) for more details.

## Framework Integration

Celery seamlessly integrates with popular Python web frameworks:

| Framework    | Integration                               |
|--------------|-------------------------------------------|
| Django       | Not needed                                |
| Pyramid      | `pyramid_celery`                           |
| Pylons       | `celery-pylons`                            |
| Flask        | Not needed                                |
| web2py       | `web2py-celery`                            |
| Tornado      | `tornado-celery`                           |
| FastAPI      | Not needed                                |

## Supported Python Versions

Celery 5.5.x supports:

*   Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
*   PyPy3.9+ (v7.3.12+)

##  Bundles

Bundles allow you to install Celery with specific dependencies for different features:
```bash
    pip install "celery[redis]"
    pip install "celery[redis,auth,msgpack]"
```
Available Bundles:  *auth, msgpack, yaml, eventlet, gevent, amqp, redis, sqs, tblib, memcache, pymemcache, cassandra, azureblockblob, s3, gcs, couchbase, arangodb, elasticsearch, riak, cosmosdbsql, zookeeper, sqlalchemy, pyro, slmq, consul, django, gcpubsub*

## Sponsors

Celery is supported by several sponsors:

*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

### Open Collective

Support the Celery project by becoming a backer or sponsor:

*   [Open Collective](https://opencollective.com/celery)

## Documentation and Resources

*   **Documentation:** [Latest Documentation](https://docs.celeryq.dev/en/latest/)
*   **Source Code:** [GitHub Repository](https://github.com/celery/celery)
*   **Issue Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)
*   **Wiki:** [Celery Wiki](https://github.com/celery/celery/wiki)
*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** `#celery` on [Libera Chat](https://libera.chat/)

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).

---