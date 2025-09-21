[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue that helps you manage asynchronous tasks in Python.**

[View the Celery Repository on GitHub](https://github.com/celery/celery)

**Key Features:**

*   **Simple to Use:** Celery is designed for ease of use and requires minimal configuration.
*   **Highly Available:** Workers and clients automatically retry in case of connection failures.
*   **Fast Performance:** Process millions of tasks per minute with sub-millisecond latency (with optimized settings).
*   **Flexible and Extensible:** Customize almost every part of Celery to fit your needs.
*   **Broad Framework Integration:** Seamless integration with popular Python frameworks like Django, Flask, and Pyramid.
*   **Supports Multiple Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Versatile Concurrency:** Supports Prefork, Eventlet, gevent, and single-threaded ("solo") concurrency models.
*   **Multiple Result Stores:** AMQP, Redis, memcached, SQLAlchemy, and more.
*   **Various Serialization Options:** Pickle, JSON, YAML, msgpack with compression and cryptographic message signing.

## Sponsors

Celery is supported by the following organizations:

*   **Blacksmith:**  (link to Blacksmith)
*   **CloudAMQP:** (link to CloudAMQP)
*   **Upstash:** (link to Upstash)
*   **Dragonfly:** (link to Dragonfly)

## What is a Task Queue?

A task queue distributes work across threads or machines, managing asynchronous tasks efficiently. Celery uses a message broker to coordinate between clients (who send tasks) and workers (who execute them). This architecture enables high availability and horizontal scalability.

## Getting Started

For first-time users or those upgrading from older Celery versions, start with these tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Installation

Install Celery using `pip`:

```bash
pip install -U Celery
```

For detailed installation instructions, including installing from source and using development versions, see the [Installation](#installation) section in the full documentation.

## Bundles

Celery offers bundles for easy installation of dependencies for specific features.
For example:

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

Available bundles include:
*   Serializers, such as `[auth]`, `[msgpack]`, `[yaml]`
*   Concurrency, such as `[eventlet]`, `[gevent]`
*   Transports and Backends, such as `[amqp]`, `[redis]`, `[sqs]`, `[gcs]`, and many more.
See the [Installation](#installation) section for a full list.

## Documentation

Comprehensive documentation is available at [latest documentation](https://docs.celeryq.dev/en/latest/), including user guides, tutorials, and an API reference.

## Support and Community

*   **Mailing List:** Join the [celery-users mailing list](https://groups.google.com/group/celery-users/) for discussions.
*   **IRC:** Chat with the community on the `#celery` channel at [Libera Chat](https://libera.chat/).
*   **Bug Tracker:** Report issues and suggestions on the [issue tracker](https://github.com/celery/celery/issues/).
*   **Wiki:** Explore the [Celery Wiki](https://github.com/celery/celery/wiki) for additional resources.

## Contributing

Celery is an open-source project, and contributions are welcome!

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).