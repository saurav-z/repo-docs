<div align="center">
  <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner">
</div>

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue for Python, designed to handle asynchronous tasks and background processing.**

[Link to Original Repo](https://github.com/celery/celery)

## Key Features

*   **Simple to Use:** Easy setup and maintenance without complex configuration files.
*   **Highly Available:** Built-in retry mechanisms for worker and client resilience.
*   **Fast Performance:** Capable of processing millions of tasks per minute with sub-millisecond latency.
*   **Flexible and Extensible:** Supports custom implementations for pools, serializers, logging, and more.
*   **Multiple Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub and more.
*   **Concurrency:** Offers prefork, eventlet, gevent, and single-threaded (solo) concurrency models.
*   **Result Stores:** Integrates with various stores like AMQP, Redis, memcached, SQLAlchemy, and more.
*   **Serialization:** Supports pickle, json, yaml, msgpack, and compression schemes like zlib and bzip2.
*   **Framework Integration:** Seamlessly integrates with popular Python frameworks like Django, Flask, and Pyramid.

## What is a Task Queue?

Task queues are essential for distributing work across threads or machines. Celery enables this by using a message broker to manage communication between clients and workers. Clients submit tasks (units of work) to the queue, which the broker then delivers to available workers. This architecture enables high availability and horizontal scaling, ideal for handling complex asynchronous workflows.

## Who Uses Celery?

Celery is perfect for anyone who needs to offload background tasks from their main application processes. Examples include:

*   **Web applications:** Running tasks such as sending emails, processing images, or updating databases without blocking user requests.
*   **Data processing:** Handling large datasets, performing complex calculations, or ETL processes.
*   **Microservices:** Enabling communication and task coordination between different services.

## Requirements

*   Python 3.8 - 3.13
*   PyPy3.9+ (v7.3.12+)
*   Message Broker: RabbitMQ, Redis (recommended), and others.

## Installation

Install Celery using pip:

```bash
pip install -U celery
```

## Bundles

Celery offers bundles to install Celery and dependencies for features. For example:

```bash
pip install "celery[redis]"
```

Available bundles:  `auth`, `msgpack`, `yaml`, `eventlet`, `gevent`, `amqp`, `redis`, `sqs`, `tblib`, `memcache`, `pymemcache`, `cassandra`, `azureblockblob`, `s3`, `gcs`, `couchbase`, `arangodb`, `elasticsearch`, `riak`, `cosmosdbsql`, `zookeeper`, `sqlalchemy`, `pyro`, `slmq`, `consul`, `django`, `gcpubsub`

## Documentation

Access the latest documentation with user guides, tutorials, and API references here:  [Celery Documentation](https://docs.celeryq.dev/en/latest/)

## Get Involved

*   **Contribute:**  Contribute to Celery's development on GitHub.
*   **Get Help:** Join the `celery-users` mailing list or the `#celery` IRC channel on Libera Chat for support.
*   **Report Bugs:**  Report issues on the GitHub issue tracker.

## Sponsors

Thank you to our sponsors!

[Blacksmith Logo](https://blacksmith.sh/)
[CloudAMQP Logo](https://www.cloudamqp.com/)
[Upstash Logo](http://upstash.com/?code=celery)
[Dragonfly Logo](https://www.dragonflydb.io/)

## Support

*   **Open Collective:**  Support Celery's development through our community-powered funding platform: [Open Collective](https://opencollective.com/celery)
*   **Tidelift:**  Commercial support and maintenance are available through the Tidelift Subscription: [Tidelift](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).