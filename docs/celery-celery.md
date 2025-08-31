[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and flexible distributed task queue that enables asynchronous task processing, perfect for improving application performance and scalability.** ([Original Repo](https://github.com/celery/celery))

## Key Features

*   **Simple to Use:** Easy to set up and maintain, with no configuration files needed.
*   **Highly Available:** Built-in automatic retries for tasks and worker resilience.
*   **Blazing Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible and Extensible:** Supports customization of almost every aspect of the system.
*   **Multiple Transports:** Works with RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Various Concurrency Models:** Offers prefork, Eventlet, gevent, and single-threaded (solo) options.
*   **Result Stores:** Supports AMQP, Redis, Memcached, SQLAlchemy, and various other databases and storage solutions.
*   **Serialization Support:** Includes pickle, JSON, YAML, and msgpack, with compression and cryptographic signing.

## What is a Task Queue?

Task queues are a crucial mechanism for distributing work across threads or machines. Celery uses a message broker to facilitate communication between clients and workers. Clients send tasks to the queue, and workers pick up the tasks to perform them.  This architecture provides high availability and the ability to scale horizontally.

## Getting Started

### Requirements

*   Python 3.8+
*   PyPy3.9+ (v7.3.12+)

### Installation

**Using pip:**

```bash
pip install -U Celery
```

**Bundles:**

Celery offers bundles for easy installation of dependencies.
*   Serializers: `celery[auth]`, `celery[msgpack]`, `celery[yaml]`
*   Concurrency: `celery[eventlet]`, `celery[gevent]`
*   Transports and Backends: (RabbitMQ, Redis, SQS, Google Pub/Sub, and more)
    *   `celery[amqp]`
    *   `celery[redis]`
    *   `celery[sqs]`
    *   `celery[gcpubsub]`
    *   ... and more

**See more details on installing bundles** [Here](https://github.com/celery/celery#bundles)

### Core Concepts
1.  **Tasks:** Units of work performed by workers.
2.  **Brokers:** Message brokers like RabbitMQ, Redis, or Amazon SQS that Celery uses to send and receive messages.
3.  **Workers:** Dedicated processes that monitor the queue for new tasks.

### First Steps

*   Check out the documentation, tutorials, and API references: [Celery Documentation](https://docs.celeryq.dev/en/latest/)
*   [Getting Started with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)

## Framework Integration

Celery integrates seamlessly with popular web frameworks:

| Framework        | Integration Package       |
| ---------------- | ------------------------- |
| Django           | Built-in                 |
| Pyramid          | pyramid_celery            |
| Pylons           | celery-pylons             |
| Flask            | Built-in                 |
| web2py           | web2py-celery             |
| Tornado          | tornado-celery            |
| FastAPI          | Built-in                  |

## Sponsors

Celery's development is supported by:

*   **Blacksmith:** (Logo) [Blacksmith](https://blacksmith.sh/)
*   **CloudAMQP:** (Logo) [CloudAMQP](https://www.cloudamqp.com/)
*   **Upstash:** (Logo) [Upstash](http://upstash.com/?code=celery)
*   **Dragonfly:** (Logo) [Dragonfly](https://www.dragonflydb.io/)

## Open Collective and Funding

Celery is community-driven and supported by donations through [Open Collective](https://opencollective.com/celery).  Your sponsorship directly supports Celery's development.

## For Enterprise

Celery is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo).

## Support and Community

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** #celery on Libera Chat
*   **Bug Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)
*   **Wiki:** [GitHub Wiki](https://github.com/celery/celery/wiki)

## Credits

*   [Contributors](https://github.com/celery/celery/graphs/contributors)
*   [Backers](https://opencollective.com/celery#backers)

## License

*   [New BSD License](https://opensource.org/licenses/BSD-3-Clause)