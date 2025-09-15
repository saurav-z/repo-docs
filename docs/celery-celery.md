[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful, easy-to-use distributed task queue that helps you schedule and execute asynchronous tasks in Python.**

*   **Version:** 5.6.0b1 (recovery)
*   **Web:** [https://docs.celeryq.dev/en/stable/index.html](https://docs.celeryq.dev/en/stable/index.html)
*   **Download:** [https://pypi.org/project/celery/](https://pypi.org/project/celery/)
*   **Source:** [https://github.com/celery/celery](https://github.com/celery/celery)

## Key Features

*   **Simple:** Easy to learn, use, and maintain, with no configuration files needed.
*   **Highly Available:** Built-in retry mechanisms and broker support for high availability.
*   **Fast:** Capable of processing millions of tasks per minute with low latency.
*   **Flexible:** Extensible for custom pool implementations, serializers, compression schemes, and more.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency:** Offers prefork, Eventlet, gevent, and single-threaded (solo) concurrency models.
*   **Result Stores:** Compatible with AMQP, Redis, memcached, SQLAlchemy, and various cloud storage solutions.
*   **Serialization:** Supports pickle, json, yaml, msgpack, and compression schemes.

## What is a Task Queue?

Task queues are used to distribute work across threads or machines. Celery allows clients to initiate tasks by putting messages on a queue, which are then picked up and processed by worker processes. This enables asynchronous task execution, increasing application performance and scalability.

## Get Started

Explore our getting started tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Framework Integration

Celery integrates seamlessly with popular Python web frameworks:

| Framework   | Integration Package |
| ----------- | ------------------- |
| Django      | N/A                 |
| Pyramid     | pyramid_celery      |
| Pylons      | celery-pylons       |
| Flask       | N/A                 |
| web2py      | web2py-celery       |
| Tornado     | tornado-celery      |
| FastAPI     | N/A                 |

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

You can install Celery with specific features (bundles):

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

See the original [README](https://github.com/celery/celery) for the full list of bundles.

##  Donations

Celery's development is fueled by community support.  Contribute via:

*   **Open Collective:** [https://opencollective.com/celery](https://opencollective.com/celery)

## Sponsors

Celery is supported by these great companies:

*   **Blacksmith:** [https://blacksmith.sh/](https://blacksmith.sh/)
*   **CloudAMQP:** [https://www.cloudamqp.com/](https://www.cloudamqp.com/)
*   **Upstash:** [http://upstash.com/?code=celery](http://upstash.com/?code=celery)
*   **Dragonfly:** [https://www.dragonflydb.io/](https://www.dragonflydb.io/)

## Getting Help

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** #celery on Libera Chat ([https://libera.chat/](https://libera.chat/))
*   **Issue Tracker:** [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)

## License

Celery is licensed under the [BSD License](https://opensource.org/licenses/BSD-3-Clause).

## Contributing

We welcome contributions! Development happens at GitHub: [https://github.com/celery/celery](https://github.com/celery/celery)

## Credits

Thanks to all [contributors](https://github.com/celery/celery/graphs/contributors) and [backers](https://opencollective.com/celery#backers)!

---

**[Return to Celery's GitHub Repository](https://github.com/celery/celery)**