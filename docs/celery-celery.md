[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue for Python, enabling asynchronous task execution and real-time processing.**

[View the original repository on GitHub](https://github.com/celery/celery)

**Key Features:**

*   **Simple to Use:** Celery is designed to be easy to learn and integrate into your Python projects, without the need for complex configuration files.
*   **Highly Available:** Built-in mechanisms for automatic retries and failover ensure tasks are completed, even in the event of connection issues.
*   **Fast Performance:** Process millions of tasks per minute with low latency.
*   **Flexible & Extensible:** Customize almost every aspect of Celery, from pool implementations to serialization and broker transports.
*   **Message Transport Support:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency Support:** Prefork, Eventlet, gevent, and single-threaded (solo) modes.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, Django ORM, and various other storage options.
*   **Serialization Options:** Supports pickle, JSON, YAML, and msgpack, along with compression and cryptographic signing.

## What is a Task Queue?

Task queues are used to distribute work across threads or machines. Celery uses a message broker to mediate between clients and workers. This allows for high availability and horizontal scaling. Celery is written in Python, and can be implemented in any language, with clients available in other languages like Node.js, PHP, Go, and Rust.

## What do I Need?

Celery version 5.5.x runs on:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

Older Python versions require older versions of Celery.

## Get Started

Read the getting started tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Framework Integration

Celery integrates easily with popular web frameworks.

## Installation

```bash
pip install -U Celery
```

Or install a bundle:

```bash
pip install "celery[redis]"
```

## Documentation

Access the full documentation: [https://docs.celeryq.dev/en/latest/](https://docs.celeryq.dev/en/latest/)

## Sponsors

Celery is supported by a community of sponsors, including:

*   **Blacksmith:** [https://blacksmith.sh/](https://blacksmith.sh/)
*   **CloudAMQP:** [https://www.cloudamqp.com/](https://www.cloudamqp.com/)
*   **Upstash:** [http://upstash.com/?code=celery](http://upstash.com/?code=celery)
*   **Dragonfly:** [https://www.dragonflydb.io/](https://www.dragonflydb.io/)

## Community

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** #celery on Libera Chat ([https://libera.chat/](https://libera.chat/))
*   **Bug Tracker:** [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)
*   **Wiki:** [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki)

## License

This software is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).