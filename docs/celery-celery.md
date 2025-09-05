[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful, easy-to-use, and flexible distributed task queue that allows you to run tasks asynchronously in Python.**

[Original Repository](https://github.com/celery/celery) | [Documentation](https://docs.celeryq.dev/en/stable/index.html) | [PyPI](https://pypi.org/project/celery/)

## Key Features

*   **Simple & Easy to Use:**  Celery is designed for ease of use and requires minimal configuration.
*   **High Availability:**  Built-in retry mechanisms and broker support ensure reliability.
*   **Fast Performance:**  Processes millions of tasks per minute with low latency.
*   **Flexible & Extensible:**  Customize almost every aspect, including pools, serializers, and transports.
*   **Multiple Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency Support:** Offers prefork, Eventlet, gevent, and single-threaded (solo) concurrency models.
*   **Multiple Result Stores:** Compatible with AMQP, Redis, memcached, SQLAlchemy, and cloud storage solutions.
*   **Serialization Support:** Supports *pickle*, *json*, *yaml*, *msgpack*, and compression options.
*   **Framework Integration:** Seamless integration with popular Python web frameworks like Django, Flask, and Pyramid.

## What is a Task Queue?

Task queues are used to distribute work across threads or machines. Celery uses message brokers to handle communication between clients and workers.

## Who is using Celery?

Celery is used by many different businesses and organizations. Here are some of the businesses that are using Celery:

*   CloudAMQP ([CloudAMQP is a RabbitMQ as a service provider](https://www.cloudamqp.com/))
*   Upstash ([Upstash offers a serverless Redis database service](http://upstash.com/?code=celery))
*   Dragonfly ([Dragonfly is a drop-in Redis replacement that cuts costs and boosts performance](https://www.dragonflydb.io/))
*   Blacksmith ([Blacksmith powers Celery](https://blacksmith.sh/))

## Getting Started

If you're new to Celery, explore these tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Installation

Install Celery using pip:

```bash
pip install -U celery
```

## Bundles

Celery includes bundles to simplify installation with specific dependencies:

*   **Serializers:**  `[auth]`, `[msgpack]`, `[yaml]`
*   **Concurrency:** `[eventlet]`, `[gevent]`
*   **Transports and Backends:**  `[amqp]`, `[redis]`, `[sqs]`, `[tblib]`, `[memcache]`,  `[pymemcache]`, `[cassandra]`, `[azureblockblob]`, `[s3]`, `[gcs]`, `[couchbase]`, `[arangodb]`, `[elasticsearch]`, `[riak]`, `[cosmosdbsql]`, `[zookeeper]`, `[sqlalchemy]`, `[pyro]`, `[slmq]`, `[consul]`, `[django]`, `[gcpubsub]`

Install bundles like this:

```bash
pip install "celery[redis]"
```

## Version Compatibility

*   **Python 3.8+:** Celery 5.5.x
*   **Python 3.7:** Celery 5.2 or earlier.
*   **Python 3.6:** Celery 5.1 or earlier.
*   **Python 2.7:** Celery 4.x series.
*   **Python 2.6:** Celery series 3.1 or earlier.
*   **Python 2.5:** Celery series 3.0 or earlier.
*   **Python 2.4:** Celery series 2.2 or earlier.

## Getting Help

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** `#celery` on Libera Chat
*   **Bug Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)

## Contributing

We welcome contributions! See the [Contributing to Celery](https://docs.celeryq.dev/en/stable/contributing.html) guide.

## Support & Sponsorship

*   **Open Collective:**  [https://opencollective.com/celery](https://opencollective.com/celery)
*   **Tidelift:** [Learn more. <https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo>`_]

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).