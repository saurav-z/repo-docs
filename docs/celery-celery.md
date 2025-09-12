# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue that enables asynchronous task execution, making your Python applications more efficient and scalable.**  [View the original repository on GitHub](https://github.com/celery/celery)

## Key Features

*   **Asynchronous Task Execution:** Offloads time-consuming tasks to the background, preventing blocking and improving responsiveness.
*   **Distributed Processing:** Distributes tasks across multiple workers on different machines for parallel processing.
*   **Message Broker Support:** Supports popular message brokers like RabbitMQ, Redis, Amazon SQS, and Google Pub/Sub.
*   **Flexible Concurrency Models:** Offers various concurrency models, including prefork, eventlet, and gevent.
*   **Multiple Result Stores:** Supports various result stores such as AMQP, Redis, memcached, SQLAlchemy, and more.
*   **Serialization Support:** Supports pickle, json, yaml, and msgpack serialization, with compression and cryptographic signing options.
*   **Framework Integration:** Seamlessly integrates with popular Python web frameworks like Django, Flask, and Pyramid.
*   **Scalability:** Designed for high availability and horizontal scaling, capable of handling millions of tasks.
*   **Ease of Use:** Simple to set up and maintain, with an active community and extensive documentation.

## What is a Task Queue?

Task queues are a mechanism to distribute work across threads or machines, with input of work handled by dedicated worker processes. Celery utilizes a message broker to mediate between clients and workers. Clients put messages on the queue, and the broker delivers these messages to workers. This architecture allows for high availability and horizontal scaling.

## Who is Celery for?

*   Developers needing to improve web application performance.
*   Applications performing resource-intensive processes such as data processing and video encoding.
*   Anyone needing to build scalable, resilient applications.

## Requirements

Celery version 5.5.x supports:

*   Python: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
*   PyPy3.9+ (v7.3.12+)

## Get Started

Refer to the `First steps with Celery`_ and `Next steps`_ tutorials.

## Sponsors

Celery is supported by these amazing sponsors:

*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

## Bundles

Use bundles to install Celery and feature dependencies:

```bash
pip install "celery[redis]"
```

Available bundles for serializers, concurrency, transports, and backends are outlined in the original README.

## Documentation

*   [Latest Documentation](https://docs.celeryq.dev/en/latest/)

## Getting Help

*   **Mailing List:** Join the `celery-users`_ mailing list for discussions.
*   **IRC:** Chat with us on IRC in the **#celery** channel on `Libera Chat`_.
*   **Issue Tracker:** Report suggestions, bugs, or issues at https://github.com/celery/celery/issues/

## Contributing

Development happens at GitHub: https://github.com/celery/celery.  Contributions are welcome!  See the `Contributing to Celery`_ section in the documentation for more information.

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).