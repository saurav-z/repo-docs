[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue for Python, enabling asynchronous task execution and background processing.**

[View the original repository on GitHub](https://github.com/celery/celery)

## Key Features

*   **Simple:** Easy to use and maintain with no configuration files required.
*   **Highly Available:** Automatically retries tasks and supports HA brokers.
*   **Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible:** Extensible for custom implementations, serializers, and more.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency:** Prefork, Eventlet, gevent, and single-threaded (solo) options.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, and various cloud storage options.
*   **Serialization:** Includes pickle, json, yaml, and msgpack with compression and cryptographic signing options.
*   **Framework Integration:** Seamless integration with Django, Pyramid, Flask, and other popular Python frameworks.

## What is a Task Queue?

Task queues are a mechanism to distribute work across threads or machines.  Celery utilizes task queues, where tasks are defined as units of work and processed by dedicated worker processes monitoring the queue.

## Getting Started

### Installation

```bash
pip install -U Celery
```

or

```bash
pip install "celery[redis]"
```

### Documentation

*   **Latest Documentation:** [https://docs.celeryq.dev/en/latest/](https://docs.celeryq.dev/en/latest/)

### Example
```python
from celery import Celery

app = Celery('hello', broker='amqp://guest@localhost//')

@app.task
def hello():
    return 'hello world'
```

## Sponsors
Celery is supported by individuals and organizations.
*   **Open Collective:** Your sponsorships help fuel Celery's development, ensuring its robustness and reliability. [Open Collective](https://opencollective.com/celery)
*   **Blacksmith:** [Blacksmith](https://blacksmith.sh/)
*   **CloudAMQP:** [CloudAMQP](https://www.cloudamqp.com/)
*   **Upstash:** [Upstash](http://upstash.com/?code=celery)
*   **Dragonfly:** [Dragonfly](https://www.dragonflydb.io/)

## Support

*   **Mailing List:** Join the [celery-users](https://groups.google.com/group/celery-users/) mailing list for discussions.
*   **IRC:** Chat with us on IRC at the [Libera Chat](https://libera.chat/) network, channel `#celery`.
*   **Bug Tracker:** Report issues and suggestions on our [GitHub issue tracker](https://github.com/celery/celery/issues/).

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).