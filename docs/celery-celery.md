<!-- Celery - Distributed Task Queue for Python -->
[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage Status](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementation](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)

# Celery: Asynchronous Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue, enabling you to execute tasks asynchronously in your Python applications.**  [Explore the original repo](https://github.com/celery/celery).

## Key Features

*   **Simple:**  Easy to learn, use, and maintain, requiring no configuration files.
*   **Highly Available:**  Workers and clients automatically retry in case of connection issues, and brokers support high availability.
*   **Fast:**  Handles millions of tasks per minute with low latency.
*   **Flexible:**  Extensible with custom implementations for almost every component.
*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency:** Supports Prefork, Eventlet, gevent, and single-threaded (solo) concurrency models.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, Google Cloud Storage and more.
*   **Serialization:** pickle, json, yaml, msgpack, zlib, bzip2 compression, and cryptographic message signing.

## What is a Task Queue?

Task queues distribute work across threads or machines. Celery uses a message broker to pass messages between clients (which initiate tasks) and workers (which execute tasks). This enables high availability and horizontal scaling.

## Getting Started

*   Install using pip:  `pip install -U Celery`
*   Refer to the [official documentation](https://docs.celeryq.dev/en/latest/) for detailed guides and tutorials.

## Framework Integration

Celery integrates seamlessly with popular web frameworks:

| Framework | Integration Method |
| --------- | ------------------ |
| Django    | Not needed         |
| Pyramid   | pyramid\_celery    |
| Pylons    | celery-pylons       |
| Flask     | Not needed         |
| web2py    | web2py-celery       |
| Tornado   | tornado-celery      |
| FastAPI   | Not needed         |

## Sponsors

Thank you to the following organizations for supporting Celery development:

*   **Blacksmith:**  [Blacksmith](https://blacksmith.sh/)
*   **CloudAMQP:**  [CloudAMQP](https://www.cloudamqp.com/)
*   **Upstash:**  [Upstash](http://upstash.com/?code=celery)
*   **Dragonfly:**  [Dragonfly](https://www.dragonflydb.io/)

## Donations and Support

*   **Open Collective:** [Open Collective](https://opencollective.com/celery) - your support fuels ongoing development.
*   **Tidelift Subscription:**  Available as part of the Tidelift Subscription for commercial support and maintenance. [Learn more](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo).

## Contributing

Contributions are welcome!  Find us on GitHub and read the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) documentation.

## Getting Help

*   **Mailing List:** Join the `celery-users`_ mailing list for discussions and support.
*   **IRC:** Chat with us on IRC in the #celery channel on Libera Chat.
*   **Bug Tracker:** Report issues at [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)
*   **Wiki:** Access the Celery Wiki on GitHub: [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki)

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).