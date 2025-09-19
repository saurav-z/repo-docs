[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue, perfect for handling asynchronous tasks in your Python applications.**  [Explore the official Celery repository](https://github.com/celery/celery).

## Key Features

*   **Simple to Use:**  Celery is designed for ease of use and maintenance, with no configuration files required, and a supportive community.
*   **Highly Available:** Workers and clients automatically retry in the event of connection failures.
*   **Fast Performance:**  Process millions of tasks per minute with sub-millisecond latency.
*   **Flexible Architecture:** Extensible components, allowing you to customize pool implementations, serializers, concurrency, and more.

## Core Capabilities

*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and others.
*   **Concurrency Models:**  Prefork, Eventlet, gevent, and single-threaded (solo) options.
*   **Result Stores:**  AMQP, Redis, memcached, SQLAlchemy, and other database and storage solutions.
*   **Serialization:**  pickle, json, yaml, msgpack, with zlib, bzip2 compression, and cryptographic message signing.

## Getting Started

### Installation

Install Celery using pip:

```bash
pip install -U Celery
```

For more detailed instructions, refer to the [Installation](#installation) section of this README.

### Learning Resources

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Framework Integration

Celery integrates seamlessly with popular Python web frameworks:

| Framework  | Integration |
| ----------- | ----------- |
| Django      | (Not needed)  |
| Pyramid     | pyramid\_celery |
| Pylons      | celery-pylons |
| Flask       | (Not needed)  |
| web2py      | web2py-celery |
| Tornado     | tornado-celery |
| FastAPI     | (Not needed)  |

## Sponsors

Celery is supported by generous sponsors:

*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

## Funding and Community

*   **Open Collective:**  Support Celery's development through [Open Collective](https://opencollective.com/celery).
*   **Mailing List:** Join the [celery-users](https://groups.google.com/group/celery-users/) mailing list for discussions.
*   **IRC:** Chat with the community in the **#celery** channel on [Libera Chat](https://libera.chat/).
*   **Bug Tracker:** Report issues and contribute at the [GitHub issue tracker](https://github.com/celery/celery/issues/).

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).  See the `LICENSE` file for details.