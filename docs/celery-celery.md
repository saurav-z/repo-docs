[![Celery Banner](https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png)](https://github.com/celery/celery)

# Celery: Distributed Task Queue for Python

**Celery is a powerful and easy-to-use distributed task queue that helps you manage asynchronous tasks in Python applications.**  [View the original repository](https://github.com/celery/celery)

**Key Features:**

*   **Simple to Use:**  Get started quickly with minimal configuration.
*   **Highly Available:**  Built-in mechanisms for retries and HA.
*   **Fast:**  Capable of processing millions of tasks per minute.
*   **Flexible:** Extensible and supports a wide range of options.
*   **Message Transports:** Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Concurrency:**  Prefork, Eventlet, gevent, and single-threaded options.
*   **Result Stores:** AMQP, Redis, memcached, SQLAlchemy, and others.
*   **Serialization:** JSON, YAML, Msgpack, and more.

**What is a Task Queue?**

Task queues are used to distribute work across threads or machines.  Celery utilizes message brokers to enable communication between clients and workers.  A client enqueues a task, the broker delivers it to a worker for execution. Celery supports high availability and horizontal scaling.

**Getting Started**

Follow our getting started tutorials:

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

**Supported Python Versions:**

*   Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
*   PyPy3.9+ (v7.3.12+)

**Framework Integration**

Celery integrates seamlessly with various Python frameworks, including Django, Pyramid, Flask, and more.

**Installation**

```bash
pip install -U Celery
```

**Bundles**

Use bundles to install celery and dependencies for certain features. Example:
```bash
pip install "celery[redis]"
pip install "celery[redis,auth,msgpack]"
```
See the full list in the original README for a description of available bundles.

**Resources**

*   [Documentation](https://docs.celeryq.dev/en/latest/)
*   [Issue Tracker](https://github.com/celery/celery/issues/)
*   [Wiki](https://github.com/celery/celery/wiki)

**Community and Support**

*   [Mailing List](https://groups.google.com/group/celery-users/)
*   [IRC Channel](https://libera.chat/)

**Sponsors**

Special thanks to the sponsors who support Celery's development.  See the original README for a list of sponsors.

**License**

Celery is licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).