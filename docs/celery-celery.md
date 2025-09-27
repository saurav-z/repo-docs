![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)

# Celery: Distributed Task Queue for Python

**Celery is an open-source, distributed task queue that enables asynchronous task processing, perfect for background jobs and real-time operations.**  [View the original repository on GitHub](https://github.com/celery/celery)

[![Build Status](https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg)](https://github.com/celery/celery/actions/workflows/python-package.yml)
[![Coverage Status](https://codecov.io/github/celery/celery/coverage.svg?branch=main)](https://codecov.io/github/celery/celery?branch=main)
[![License](https://img.shields.io/pypi/l/celery.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Wheel](https://img.shields.io/pypi/wheel/celery.svg)](https://pypi.org/project/celery/)
[![Semgrep Security](https://img.shields.io/badge/semgrep-security-green.svg)](https://go.semgrep.dev/home)
[![Python Versions](https://img.shields.io/pypi/pyversions/celery.svg)](https://pypi.org/project/celery/)
[![Python Implementations](https://img.shields.io/pypi/implementation/celery.svg)](https://pypi.org/project/celery/)
[![Open Collective Backers](https://opencollective.com/celery/backers/badge.svg)](https://opencollective.com/celery#backers)
[![Open Collective Sponsors](https://opencollective.com/celery/sponsors/badge.svg)](https://opencollective.com/celery#sponsors)
[![Downloads](https://pepy.tech/badge/celery)](https://pepy.tech/project/celery)
[DeepWiki](https://deepwiki.com/celery/celery)

**Key Features:**

*   **Asynchronous Task Execution:** Offloads time-consuming tasks to the background, improving application responsiveness.
*   **Distributed Processing:** Distributes tasks across multiple workers and machines for scalability.
*   **Message Broker Support:** Integrates seamlessly with popular message brokers like RabbitMQ, Redis, and others.
*   **Simple to Use:** Easy to set up and configure, with minimal configuration files required.
*   **Highly Available:** Provides automatic retries and supports HA brokers for robust operation.
*   **Fast Performance:** Processes millions of tasks per minute with low latency.
*   **Flexible and Extensible:**  Customizable with support for custom pool implementations, serializers, compression schemes, and more.
*   **Framework Integration:**  Provides easy integration with popular Python web frameworks like Django, Flask, and Pyramid.

## What's a Task Queue?

Task queues are the backbone of asynchronous task processing, distributing work across threads or machines.  Celery utilizes task queues to manage units of work, which are then processed by dedicated worker processes monitoring the queue for new tasks to perform. Celery communicates using messages, facilitating interaction between clients and workers through a message broker.

## Getting Started

To get started with Celery, check out these tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Sponsors

Celery's development is supported by the following sponsors:

### Blacksmith

[![Blacksmith Logo](docs/images/blacksmith-logo-white-on-black.svg)](https://blacksmith.sh/)

### CloudAMQP

[![CloudAMQP Logo](docs/images/cloudamqp-logo-lightbg.svg)](https://www.cloudamqp.com/)

### Upstash

[![Upstash Logo](https://upstash.com/logo/upstash-dark-bg.svg)](http://upstash.com/?code=celery)

### Dragonfly

[![Dragonfly Logo](https://github.com/celery/celery/raw/main/docs/images/dragonfly.svg)](https://www.dragonflydb.io/)

## Donations

Celery's development is powered by community funding.  Your contributions directly support improvements and maintenance.

### Open Collective

[![Open Collective Logo](https://opencollective.com/static/images/opencollectivelogo-footer-n.svg)](https://opencollective.com/celery)

## For Enterprise

Celery is available as part of the Tidelift Subscription.  Learn more at the Tidelift website: [https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## What do I need?

Celery version 5.5.x runs on:

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

Celery is typically used with a message broker.  RabbitMQ and Redis are feature-complete options, but experimental support exists for other solutions, including SQLite for local development.

## It supports...

*   **Message Transports:** RabbitMQ, Redis, Amazon SQS, Google Pub/Sub
*   **Concurrency:** Prefork, Eventlet, gevent, single threaded (``solo``)
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, Google Cloud Storage
*   **Serialization:** pickle, json, yaml, msgpack, zlib, bzip2 compression, and cryptographic message signing

## Framework Integration

Celery provides easy integration with popular Python web frameworks:

*   Django
*   Pyramid
*   Pylons
*   Flask
*   web2py
*   Tornado
*   FastAPI

## Documentation

The [latest documentation](https://docs.celeryq.dev/en/latest/) is hosted on Read the Docs, offering user guides, tutorials, and an API reference.

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

### Bundles

Celery bundles simplify installation by including dependencies for specific features.

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

Available bundles:

*   Serializers: `auth`, `msgpack`, `yaml`
*   Concurrency: `eventlet`, `gevent`
*   Transports and Backends: `amqp`, `redis`, `sqs`, `tblib`, `memcache`, `pymemcache`, `cassandra`, `azureblockblob`, `s3`, `gcs`, `couchbase`, `arangodb`, `elasticsearch`, `riak`, `cosmosdbsql`, `zookeeper`, `sqlalchemy`, `pyro`, `slmq`, `consul`, `django`, `gcpubsub`

### Downloading and installing from source

Instructions are available in the original README.

### Using the development version

Instructions are available in the original README.

## Getting Help

### Mailing List

Join the [celery-users mailing list](https://groups.google.com/group/celery-users/) for discussions.

### IRC

Chat with the community on IRC in the **#celery** channel on the [Libera Chat](https://libera.chat/) network.

### Bug Tracker

Report issues and suggestions at the [issue tracker](https://github.com/celery/celery/issues/).

## Credits

### Contributors

Contribute to Celery on [GitHub](https://github.com/celery/celery).  See the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) section in the documentation.

[![Open Collective Contributors](https://opencollective.com/celery/contributors.svg?width=890&button=false)](https://github.com/celery/celery/graphs/contributors)

### Backers

Thank you to all our backers! 🙏
[![Open Collective Backers](https://opencollective.com/celery/backers.svg?width=890)](https://opencollective.com/celery#backers)

## License

Licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).