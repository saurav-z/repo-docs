[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful, easy-to-use, and flexible distributed task queue that helps you process tasks asynchronously.**

[Get Started with Celery on GitHub](https://github.com/celery/celery)

## Key Features

*   **Simple to Use:**  Easy to learn and integrate into your Python projects.
*   **Highly Available:** Built-in retry mechanisms and broker support for high availability.
*   **Fast Performance:** Capable of processing millions of tasks per minute.
*   **Flexible and Extensible:**  Customize nearly every aspect of Celery to fit your needs.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency Models:** Prefork, Eventlet, gevent, and single-threaded (solo).
*   **Result Stores:**  Supports a wide array of stores, including AMQP, Redis, Memcached, SQLAlchemy, and cloud storage options.
*   **Serialization:** Offers multiple serialization options: pickle, json, yaml, msgpack, and compression (zlib, bzip2) and cryptographic message signing.

## What is a Task Queue?

Task queues are a fundamental tool for distributing work across threads or machines. Celery facilitates this by enabling clients to submit tasks to a queue, where dedicated worker processes pick them up and execute them. This architecture allows for efficient processing of background tasks, improving application responsiveness and scalability.

## Get Started

*   **First Steps with Celery:** [https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   **Next Steps:** [https://docs.celeryq.dev/en/stable/getting-started/next-steps.html](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Framework Integration

Celery seamlessly integrates with popular Python web frameworks:

| Framework     | Integration                               |
| ------------- | ----------------------------------------- |
| Django        | Not needed                                |
| Pyramid       | pyramid_celery                           |
| Pylons        | celery-pylons                            |
| Flask         | Not needed                                |
| web2py        | web2py-celery                            |
| Tornado       | tornado-celery                           |
| FastAPI       | Not needed                                |

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

### Bundles

Use bundles to install Celery with dependencies for specific features.

```bash
pip install "celery[redis]"
```

Available bundles:

*   **Serializers:** `auth`, `msgpack`, `yaml`
*   **Concurrency:** `eventlet`, `gevent`
*   **Transports and Backends:**  `amqp`, `redis`, `sqs`, `tblib`, `memcache`, `pymemcache`, `cassandra`, `azureblockblob`, `s3`, `gcs`, `couchbase`, `arangodb`, `elasticsearch`, `riak`, `cosmosdbsql`, `zookeeper`, `sqlalchemy`, `pyro`, `slmq`, `consul`, `django`, `gcpubsub`

## Resources

*   **Documentation:** [https://docs.celeryq.dev/en/latest/](https://docs.celeryq.dev/en/latest/)
*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC Channel:** `#celery` on Libera Chat
*   **Bug Tracker:** [https://github.com/celery/celery/issues/](https://github.com/celery/celery/issues/)
*   **Wiki:** [https://github.com/celery/celery/wiki](https://github.com/celery/celery/wiki)

## Community & Support

Join the community to get help and discuss Celery:

*   **Mailing List:** Join the `celery-users`_ mailing list for discussions about usage, development, and the future of Celery.
*   **IRC:** Chat with us on IRC. The **#celery** channel is located at the `Libera Chat`_ network.

## Sponsors

Celery's development is supported by:

*   [Open Collective](https://opencollective.com/celery) - Community-powered funding
*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

<!-- Removed the following as it is not relevant to a general user of Celery -->
<!-- ## For enterprise

Available as part of the Tidelift Subscription.

The maintainers of ``celery`` and thousands of other packages are working with Tidelift to deliver commercial support and maintenance for the open source dependencies you use to build your applications. Save time, reduce risk, and improve code health, while paying the maintainers of the exact dependencies you use. `Learn more. <https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo>`_
 -->

## License

This software is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).

---

[Visit the Celery GitHub Repository](https://github.com/celery/celery)
```
Key improvements and optimizations:

*   **SEO-Optimized Hook:**  A compelling one-sentence hook is added at the beginning.
*   **Clear Headings:** Organized with clear, descriptive headings for better readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points for easy scanning of key features.
*   **Concise Descriptions:**  Descriptions are more concise and informative.
*   **Removed Irrelevant Content:** Removed enterprise section and unnecessary introductory information.
*   **Clear Calls to Action:**  Includes "Get Started" links for ease of use.
*   **Formatting for Readability:** Consistent use of bolding and markdown for enhanced readability.
*   **Simplified Installation:** The installation section and available bundles are summarized for clarity.
*   **Focus on Core Information:** The rewritten README focuses on essential information for the user.
*   **Backlink:** Added a backlink to the original repository at the end.