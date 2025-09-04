<div align="center">
  <img src="https://docs.celeryq.dev/en/latest/_images/celery-banner-small.png" alt="Celery Banner" width="600">
  <br>
  <a href="https://github.com/celery/celery"><b>Celery: Distributed Task Queue for Python</b></a>
  <br>
  <a href="https://pypi.org/project/celery/">
    <img src="https://img.shields.io/pypi/v/celery.svg" alt="PyPI">
  </a>
  <a href="https://github.com/celery/celery/actions/workflows/python-package.yml">
    <img src="https://github.com/celery/celery/actions/workflows/python-package.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://codecov.io/github/celery/celery?branch=main">
    <img src="https://codecov.io/github/celery/celery/coverage.svg?branch=main" alt="Coverage">
  </a>
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/pypi/l/celery.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/celery/">
      <img src="https://img.shields.io/pypi/wheel/celery.svg" alt="Wheel">
  </a>
  <a href="https://go.semgrep.dev/home">
      <img src="https://img.shields.io/badge/semgrep-security-green.svg" alt="Semgrep Security">
  </a>
  <a href="https://pypi.org/project/celery/">
      <img src="https://img.shields.io/pypi/pyversions/celery.svg" alt="Python Versions">
  </a>
  <a href="https://pypi.org/project/celery/">
      <img src="https://img.shields.io/pypi/implementation/celery.svg" alt="Python Implementations">
  </a>
  <a href="https://pepy.tech/project/celery">
      <img src="https://pepy.tech/badge/celery" alt="Downloads">
  </a>
</div>

## Celery: Simplify Asynchronous Tasks in Python

Celery is a powerful and easy-to-use distributed task queue that allows you to execute tasks asynchronously, manage jobs, and build scalable applications. This lets you defer processing, schedule tasks, and handle background operations efficiently, so your web app stays responsive.

**Key Features:**

*   **Simple to Use**:  Easy setup with minimal configuration and a friendly community.
*   **Highly Available**:  Automatically retries in case of connection failures, with broker support for high availability.
*   **Fast Performance**:  Processes millions of tasks per minute with low latency.
*   **Flexible and Extensible**: Customize almost every aspect of Celery to fit your needs.
*   **Multiple Message Transports**: Supports RabbitMQ, Redis, Amazon SQS, Google Pub/Sub, and more.
*   **Multiple Concurrency Models**: Supports prefork, eventlet, gevent, and single-threaded (solo).
*   **Multiple Result Stores**: Supports AMQP, Redis, memcached, SQLAlchemy, Django ORM, Apache Cassandra, IronCache, Elasticsearch, Google Cloud Storage, and more.
*   **Serialization**:  Uses pickle, json, yaml, msgpack. Includes zlib, bzip2 compression and cryptographic message signing.

**Get Started**

*   [First steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
*   [Next steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)

**What is a Task Queue?**
Task queues are used to distribute work across threads or machines. Celery communicates via messages using a broker to mediate between clients and workers. This enables high availability and horizontal scaling.

**Requirements**

*   Python (3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
*   PyPy3.9+ (v7.3.12+)

**Sponsors**

*   **Blacksmith**
    <br>
    <a href="https://blacksmith.sh/">
        <img src="./docs/images/blacksmith-logo-white-on-black.svg" alt="Blacksmith logo" width="240">
    </a>
*   **CloudAMQP**
    <br>
    <a href="https://www.cloudamqp.com/">
        <img src="./docs/images/cloudamqp-logo-lightbg.svg" alt="CloudAMQP logo" width="240">
    </a>
*   **Upstash**
    <br>
    <a href="http://upstash.com/?code=celery">
        <img src="https://upstash.com/logo/upstash-dark-bg.svg" alt="Upstash logo" width="200">
    </a>
*   **Dragonfly**
    <br>
    <a href="https://www.dragonflydb.io/">
        <img src="https://github.com/celery/celery/raw/main/docs/images/dragonfly.svg" alt="Dragonfly logo" width="150">
    </a>

**For enterprise**

Available as part of the Tidelift Subscription.
[Learn more.](https://tidelift.com/subscription/pkg/pypi-celery?utm_source=pypi-celery&utm_medium=referral&utm_campaign=enterprise&utm_term=repo)

**Installation**

```bash
pip install -U Celery
```

Or using bundles:
```bash
pip install "celery[redis]"
```
**Framework Integration**

Celery is easy to integrate with web frameworks.

| Framework       | Integration  |
| --------------- | ------------ |
| Django          | not needed   |
| Pyramid         | pyramid_celery |
| Pylons          | celery-pylons  |
| Flask           | not needed   |
| web2py          | web2py-celery |
| Tornado         | tornado-celery |
| FastAPI         | not needed   |

**Documentation**
Find the latest documentation at [Read The Docs](https://docs.celeryq.dev/en/latest/)

**Get Help**

*   [Mailing List](https://groups.google.com/group/celery-users/)
*   [IRC](https://libera.chat/) - Channel #celery on Libera Chat
*   [Bug Tracker](https://github.com/celery/celery/issues/)
*   [Wiki](https://github.com/celery/celery/wiki)

**Credits**

*   [Contributors](https://github.com/celery/celery/graphs/contributors)
    <br>
    <img src="https://opencollective.com/celery/contributors.svg?width=890&button=false" alt="Contributors">
*   [Backers](https://opencollective.com/celery#backers)
    <br>
    <img src="https://opencollective.com/celery/backers.svg?width=890" alt="Backers">

*   [Become a backer](https://opencollective.com/celery#backer)
    <br>
    <img src="https://opencollective.com/celery/backers/badge.svg" alt="Backers on Open Collective">

**License**

Licensed under the [New BSD License](https://opensource.org/licenses/BSD-3-Clause).

**Find the original repo [here](https://github.com/celery/celery).**