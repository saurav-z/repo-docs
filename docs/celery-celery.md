# Celery: Distributed Task Queue for Python

Celery is a powerful and flexible distributed task queue for Python applications, enabling asynchronous task processing and background jobs. **Streamline your application's performance and scalability with Celery, the leading choice for Python task queues.** ([Original Repo](https://github.com/celery/celery))

## Key Features

*   **Simple:** Easy to use and maintain, with no configuration files required.
*   **Highly Available:** Built-in retry mechanisms and support for HA brokers ensure tasks are processed even with connection interruptions.
*   **Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible:** Extensible architecture allowing customization of almost every part, including custom pools, serializers, transports, and more.
*   **Message Transports:** Supports popular message brokers like RabbitMQ, Redis, Amazon SQS, and Google Pub/Sub.
*   **Concurrency:** Offers multiple concurrency options, including prefork, eventlet, gevent, and single threaded (``solo``).
*   **Result Stores:** Provides various result store options, including AMQP, Redis, memcached, SQLAlchemy, and more.
*   **Serialization:** Supports multiple serialization formats: *pickle*, *json*, *yaml*, *msgpack* with *zlib* and *bzip2* compression, and cryptographic message signing.

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

Or install specific bundles for additional functionality.

### Bundles

Celery provides install bundles to support specific features:

*   **Serializers:** `[auth]`, `[msgpack]`, `[yaml]`
*   **Concurrency:** `[eventlet]`, `[gevent]`
*   **Transports and Backends:** `[amqp]`, `[redis]`, `[sqs]`, `[tblib]`, `[memcache]`, `[pymemcache]`, `[cassandra]`, `[azureblockblob]`, `[s3]`, `[gcs]`, `[couchbase]`, `[arangodb]`, `[elasticsearch]`, `[riak]`, `[cosmosdbsql]`, `[zookeeper]`, `[sqlalchemy]`, `[pyro]`, `[slmq]`, `[consul]`, `[django]`, `[gcpubsub]`

## Framework Integration

Celery integrates seamlessly with popular Python web frameworks:

*   Django (no additional packages needed)
*   Pyramid (`pyramid_celery`)
*   Pylons (`celery-pylons`)
*   Flask (no additional packages needed)
*   web2py (`web2py-celery`)
*   Tornado (`tornado-celery`)
*   FastAPI (no additional packages needed)

## Getting Started

Refer to the following resources to get up and running with Celery:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Resources

*   [Documentation](https://docs.celeryq.dev/en/latest/)
*   [Mailing List](https://groups.google.com/group/celery-users/)
*   [IRC Channel](https://libera.chat/) at #celery
*   [Bug Tracker](https://github.com/celery/celery/issues/)
*   [Wiki](https://github.com/celery/celery/wiki)

## Sponsors

Celery is supported by generous sponsors.  Learn more about supporting Celery's development by becoming a backer or sponsor.

*   [Open Collective](https://opencollective.com/celery)
*   Blacksmith
*   CloudAMQP
*   Upstash
*   Dragonfly

## Contributing

Development happens at GitHub: [https://github.com/celery/celery](https://github.com/celery/celery)

Thank you to all the [contributors](https://github.com/celery/celery/graphs/contributors).

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).