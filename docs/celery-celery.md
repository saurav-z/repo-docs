[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

**Celery: The Distributed Task Queue for Python** - easily manage asynchronous tasks and background jobs in your applications.

[View the Celery GitHub Repository](https://github.com/celery/celery)

**Key Features:**

*   **Distributed Task Execution:** Distribute work across multiple workers and machines.
*   **Asynchronous Task Processing:** Execute tasks in the background, improving application responsiveness.
*   **Message Broker Support:** Integrates with popular message brokers like RabbitMQ, Redis, and others.
*   **Flexible Configuration:** Extensible and customizable for a wide range of use cases.
*   **High Availability:** Designed for reliability with automatic retries and support for primary/primary or primary/replica replication.
*   **Scalability:** Supports horizontal scaling to handle increasing workloads.
*   **Broad Language Support:** While written in Python, supports integrations with other languages.
*   **Framework Integration:** Seamlessly integrates with popular Python frameworks like Django, Flask, and more.
*   **Result Stores:** Multiple options for storing task results, including databases, caches, and cloud storage.
*   **Active Community:** Benefit from a supportive community and extensive documentation.

**What is a Task Queue?**

Task queues are a vital component for distributing workloads in applications. They allow you to offload resource-intensive tasks from your main application thread to dedicated worker processes. This leads to improved performance, better responsiveness, and increased overall application stability.

**What's New in Celery 5.5.3 (immunity)?**

*   See the [official Celery documentation](https://docs.celeryq.dev/en/stable/index.html) for more information.

**Getting Started**

Refer to the comprehensive documentation for more information:
*   [Celery Documentation](https://docs.celeryq.dev/en/stable/index.html)
*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

**Requirements**

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

**Installation**

Install Celery using pip:

```bash
pip install -U Celery
```

**Bundles**

Celery offers bundles to install dependencies for specific features. For example, to install Redis support:

```bash
pip install "celery[redis]"
```

Available bundles:

*   `[auth]`
*   `[msgpack]`
*   `[yaml]`
*   `[eventlet]`
*   `[gevent]`
*   `[amqp]`
*   `[redis]`
*   `[sqs]`
*   `[tblib]`
*   `[memcache]`
*   `[pymemcache]`
*   `[cassandra]`
*   `[azureblockblob]`
*   `[s3]`
*   `[gcs]`
*   `[couchbase]`
*   `[arangodb]`
*   `[elasticsearch]`
*   `[riak]`
*   `[cosmosdbsql]`
*   `[zookeeper]`
*   `[sqlalchemy]`
*   `[pyro]`
*   `[slmq]`
*   `[consul]`
*   `[django]`
*   `[gcpubsub]`

**Framework Integration**

| Framework | Integration   |
| --------- | ------------- |
| Django    | not needed    |
| Pyramid   | pyramid_celery|
| Pylons    | celery-pylons |
| Flask     | not needed    |
| web2py    | web2py-celery |
| Tornado   | tornado-celery|
| FastAPI   | not needed    |

**Sponsors**

*   [Blacksmith](https://blacksmith.sh/)
*   [CloudAMQP](https://www.cloudamqp.com/)
*   [Upstash](http://upstash.com/?code=celery)
*   [Dragonfly](https://www.dragonflydb.io/)

**Donations & Support**

*   **Open Collective:** Support Celery's development via [Open Collective](https://opencollective.com/celery).
*   **Tidelift Subscription:** Enterprise support available via [Tidelift](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo).

**Getting Help**

*   **Mailing List:** [celery-users](https://groups.google.com/group/celery-users/)
*   **IRC:** `#celery` on Libera Chat ([Libera Chat](https://libera.chat/))
*   **Bug Tracker:** [GitHub Issues](https://github.com/celery/celery/issues/)
*   **Wiki:** [GitHub Wiki](https://github.com/celery/celery/wiki)

**License**

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).

**Credits**

Development of `celery` happens at GitHub: https://github.com/celery/celery