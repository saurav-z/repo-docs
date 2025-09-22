[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and flexible distributed task queue for Python that helps you manage asynchronous tasks efficiently.**

[**View the original repository on GitHub**](https://github.com/celery/celery)

Key Features:

*   **Simple to Use:** Easy to set up and maintain, minimizing configuration files.
*   **Highly Available:** Automatically retries tasks and supports High Availability (HA) configurations.
*   **Fast Performance:** Capable of processing millions of tasks per minute with low latency.
*   **Flexible and Extensible:** Customizable with support for custom pool implementations, serializers, and more.
*   **Supports Multiple Transports:** Works with RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Versatile Concurrency Models:** Supports Prefork, Eventlet, gevent, and single-threaded (solo) modes.
*   **Various Result Stores:**  Offers integration with AMQP, Redis, memcached, SQLAlchemy, and more.
*   **Flexible Serialization:** Supports pickle, json, yaml, msgpack with zlib and bzip2 compression, and cryptographic message signing.

## Get Started

To learn more about Celery:

*   **Documentation:** [https://docs.celeryq.dev/en/stable/](https://docs.celeryq.dev/en/stable/)
*   **First Steps:** [https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   **Next Steps:** [https://docs.celeryq.dev/en/stable/getting-started/next-steps.html](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Installation

Install Celery using `pip`:

```bash
pip install -U Celery
```

or with specific bundles:

```bash
pip install "celery[redis]"
```

```bash
pip install "celery[redis,auth,msgpack]"
```

Available Bundles:
*   **Serializers**: `auth`, `msgpack`, `yaml`.
*   **Concurrency**: `eventlet`, `gevent`.
*   **Transports and Backends**: `amqp`, `redis`, `sqs`, `tblib`, `memcache`, `pymemcache`, `cassandra`, `azureblockblob`, `s3`, `gcs`, `couchbase`, `arangodb`, `elasticsearch`, `riak`, `cosmosdbsql`, `zookeeper`, `sqlalchemy`, `pyro`, `slmq`, `consul`, `django`, `gcpubsub`.

For more detailed installation instructions, see the [Installation](#installation) section of the documentation.

## Framework Integration

Celery integrates seamlessly with popular web frameworks:

| Framework     | Integration                                    |
|---------------|------------------------------------------------|
| Django        | Not Needed                                     |
| Pyramid       | `pyramid_celery`                               |
| Pylons        | `celery-pylons`                                |
| Flask         | Not Needed                                     |
| web2py        | `web2py-celery`                                |
| Tornado       | `tornado-celery`                               |
| FastAPI       | Not Needed                                     |

## Sponsors

Celery's development is supported by amazing sponsors:

*   **Blacksmith:** [https://blacksmith.sh/](https://blacksmith.sh/)
*   **CloudAMQP:** [https://www.cloudamqp.com/](https://www.cloudamqp.com/)
*   **Upstash:** [http://upstash.com/?code=celery](http://upstash.com/?code=celery)
*   **Dragonfly:** [https://www.dragonflydb.io/](https://www.dragonflydb.io/)

## Donations

Support the Celery project through:

*   **Open Collective:** [https://opencollective.com/celery](https://opencollective.com/celery)

## Resources

*   **Documentation:** [https://docs.celeryq.dev/en/stable/](https://docs.celeryq.dev/en/stable/)
*   **Bug Tracker:** [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)
*   **Wiki:** [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki)
*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** #celery on Libera Chat

## License

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).

```