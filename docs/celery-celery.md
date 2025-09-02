[![Celery Logo](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue for Python, designed to handle asynchronous tasks, scheduling, and real-time processing with efficiency and flexibility.**

[View the original repository on GitHub](https://github.com/celery/celery)

## Key Features

*   **Simple to Use:** Easy setup and maintenance, no configuration files needed.
*   **Highly Available:** Workers and clients automatically retry in case of connection loss, with support for HA brokers.
*   **Fast Performance:** Processes millions of tasks per minute with low latency.
*   **Flexible and Extensible:**  Customize almost every part of Celery to fit your needs.
*   **Multiple Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Diverse Concurrency Models:** Includes Prefork, Eventlet, gevent, and single-threaded options.
*   **Result Stores:**  Offers various result stores, including AMQP, Redis, Memcached, SQLAlchemy, and cloud storage options.
*   **Serialization Support:** Supports *pickle*, *json*, *yaml*, *msgpack* serializers, *zlib*, *bzip2* compression, and cryptographic message signing.
*   **Framework Integrations:** Easy integration with popular Python web frameworks like Django, Flask, and Pyramid.

## Getting Started

### What is a Task Queue?

Task queues distribute work across threads or machines.

A task queue's input is a unit of work, called a task, dedicated worker
processes then constantly monitor the queue for new work to perform.

Celery communicates via messages, usually using a broker
to mediate between clients and workers. To initiate a task a client puts a
message on the queue, the broker then delivers the message to a worker.

A Celery system can consist of multiple workers and brokers, giving way
to high availability and horizontal scaling.

### Requirements

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

For older versions of Python, use older Celery versions.

### Installation

Install Celery using pip:

```bash
pip install -U Celery
```

or install with bundles:

```bash
pip install "celery[redis]"
```

### Getting Started Tutorials

*   `First steps with Celery`:  [https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   `Next steps`: [https://docs.celeryq.dev/en/stable/getting-started/next-steps.html](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Sponsors

Celery is supported by:

*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

## Community & Support

### Documentation

[Latest documentation](https://docs.celeryq.dev/en/latest/) is hosted at Read The Docs, containing user guides, tutorials, and an API reference.

### Help

*   **Mailing list:** `celery-users` ([https://groups.google.com/group/celery-users/](https://groups.google.com/group/celery-users/))
*   **IRC:**  \#celery on Libera Chat ([https://libera.chat/](https://libera.chat/))
*   **Bug Tracker:**  [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)
*   **Wiki:** [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki)

## Contributing

Development of `celery` happens at GitHub: [https://github.com/celery/celery](https://github.com/celery/celery)

Read the `Contributing to Celery` section in the documentation. ([https://docs.celeryq.dev/en/stable/contributing.html](https://docs.celeryq.dev/en/stable/contributing.html))

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).

---