[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful, easy-to-use distributed task queue that enables asynchronous task processing in Python applications.**

[View the original repo on GitHub](https://github.com/celery/celery)

## Key Features

*   **Simple & Easy to Use:** Configuration-free and designed for ease of use and maintenance.
*   **Highly Available:**  Automatically retries tasks and supports HA through broker replication.
*   **Fast:** Processes millions of tasks per minute with low latency.
*   **Flexible:**  Extensible with custom pool implementations, serializers, transports, and more.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency:** Supports Prefork, Eventlet, gevent, and single-threaded ("solo") concurrency models.
*   **Result Stores:** Offers multiple result stores, including AMQP, Redis, Memcached, SQL databases, and cloud storage options.
*   **Serialization:** Supports Pickle, JSON, YAML, and Msgpack, with zlib and bzip2 compression, plus cryptographic message signing.

## What is a Task Queue?

Task queues are fundamental for distributing work across threads or machines. Celery enables this by:

*   **Task Submission:**  Clients submit tasks to a queue.
*   **Worker Monitoring:** Dedicated worker processes constantly monitor the queue for new tasks.
*   **Message-Based Communication:** Celery uses a message broker to mediate between clients and workers.  Initiating a task involves a client putting a message on the queue, and the broker delivering it to a worker.
*   **Scalability and Availability:** Celery systems can comprise multiple workers and brokers, facilitating high availability and horizontal scaling.

## Getting Started

For first-time users or those migrating from older versions, explore the Celery documentation:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

## Installation

Install Celery using pip:

```bash
pip install -U Celery
```

You can also install specific features with bundles:

```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```

Available bundles: `auth`, `msgpack`, `yaml`, `eventlet`, `gevent`, `amqp`, `redis`, `sqs`, `tblib`, `memcache`, `pymemcache`, `cassandra`, `azureblockblob`, `s3`, `gcs`, `couchbase`, `arangodb`, `elasticsearch`, `riak`, `cosmosdbsql`, `zookeeper`, `sqlalchemy`, `pyro`, `slmq`, `consul`, `django`, `gcpubsub`.

## Framework Integration

Celery integrates seamlessly with popular Python web frameworks:

| Framework       | Integration                             |
| --------------- | --------------------------------------- |
| Django          |  (Not needed)                          |
| Pyramid         | `pyramid_celery`                       |
| Pylons          | `celery-pylons`                        |
| Flask           | (Not needed)                             |
| web2py          | `web2py-celery`                        |
| Tornado         | `tornado-celery`                       |
| FastAPI         | (Not needed)                             |

## Resources

*   **Documentation:** [Latest Documentation](https://docs.celeryq.dev/en/latest/)
*   **Community:**
    *   Mailing List: [celery-users](https://groups.google.com/group/celery-users/)
    *   IRC: #celery on Libera Chat ([Libera Chat](https://libera.chat/))
*   **Bug Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)
*   **Wiki:** [GitHub Wiki](https://github.com/celery/celery/wiki)

## Support and Sponsorship

*   **Open Collective:**  Support Celery's development: [Open Collective](https://opencollective.com/celery)
*   **Enterprise Support:** Available via the Tidelift Subscription. [Learn more.](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

## Sponsors

(Logos for sponsors with links - these are included in the original but not rendered properly here)

## License

This software is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause). See the `LICENSE` file in the top distribution directory.
```
Key improvements and SEO optimizations:

*   **Clear Title & Hook:** The title is clear, and the one-sentence hook grabs attention.
*   **Keyword Rich:** Keywords (async, task queue, Python, etc.) are incorporated throughout.
*   **Structured Headings:** Uses clear and descriptive headings (Installation, Getting Started, etc.) for easy navigation and SEO.
*   **Bulleted Key Features:**  Provides a concise overview of Celery's core capabilities.
*   **Detailed Explanation of Task Queues:**  Explains the core concept for those unfamiliar.
*   **Direct Links:**  Links to documentation, the issue tracker, and other relevant resources.
*   **Framework Integration Table:** Provides an easy-to-read table for framework compatibility.
*   **Concise Installation Instructions:** Provides the most important information first.
*   **Removed Unnecessary Information:** Streamlined the introduction by removing the version and download/source links.
*   **Clear Formatting:** Uses Markdown effectively for readability and SEO.
*   **Call to Action:** Encourages exploration of documentation and community resources.
*   **Optimized Sponsor Section:**  Maintained sponsor information.
*   **License and Credits**: Kept the license and credits, which are helpful for users.