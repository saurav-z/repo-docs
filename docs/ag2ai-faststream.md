# FastStream: Effortlessly Build Modern, Data-Driven Microservices 🚀

**Simplify your event stream integration and build robust microservices with FastStream, the Python framework that handles the complexities of message queues for you.**

---

<p align="center">
  <a href="https://trendshift.io/repositories/3043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3043" alt="ag2ai%2Ffaststream | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
  <br/>
  <br/>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main" alt="Test Passing"/>
  </a>

  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream" target="_blank">
      <img src="https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg" alt="Coverage"/>
  </a>

  <a href="https://www.pepy.tech/projects/faststream" target="_blank">
    <img src="https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/faststream" target="_blank">
    <img src="https://img.shields.io/pypi/v/faststream?label=PyPI" alt="Package version"/>
  </a>

  <a href="https://pypi.org/project/faststream" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/faststream.svg" alt="Supported Python versions"/>
  </a>
  <br/>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg" alt="CodeQL"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg" alt="Dependency Review"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/ag2ai/faststream.svg" alt="License"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md" target="_blank">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Code of Conduct"/>
  </a>
  <br/>

  <a href="https://discord.gg/qFm6aSqq59" target="_blank">
      <img alt="Discord" src="https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN"/>
  </a>

  <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>

  <a href="https://t.me/python_faststream" target="_blank">
    <img alt="Telegram" src="https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU"/>
  </a>
  <br/>

  <a href="https://gurubase.io/g/faststream" target="_blank">
    <img src="https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF" alt="Gurubase"/>
  </a>
</p>

---

## Key Features

*   **Multi-Broker Support:** Seamlessly integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Leverage Pydantic for robust data validation, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Generate API documentation automatically for easy service integration.
*   **Intuitive Development Experience:** Benefit from full-typed editor support, catching errors early in the development cycle.
*   **Dependency Injection:** Manage service dependencies effectively with FastStream's built-in DI system.
*   **Simplified Testing:** Utilize in-memory tests to speed up CI/CD pipelines and ensure reliability.
*   **Extensible Architecture:** Use extensions for lifespans, custom serialization, and middleware to customize your application.
*   **Framework Compatibility:**  Integrate FastStream with any HTTP framework, with dedicated support for [FastAPI](https://fastapi.tiangolo.com/).

---

**Ready to streamline your microservice development? Explore the full potential of FastStream on [GitHub](https://github.com/ag2ai/faststream)!**

---

## Why Choose FastStream?

FastStream is a modern Python framework designed for building efficient, scalable, and maintainable microservices that utilize event streams. It simplifies the complexities of working with message queues, making it easier for developers of all skill levels to build data-driven applications.

## Getting Started

### Installation

Install FastStream with your preferred broker dependencies:

```bash
pip install 'faststream[kafka]'  # For Kafka
pip install 'faststream[rabbit]' # For RabbitMQ
pip install 'faststream[nats]'   # For NATS
pip install 'faststream[redis]'  # For Redis
```

*Note: FastStream uses PydanticV2, but it will work correctly with PydanticV1 if your platform has no Rust support*

### Basic Usage

FastStream utilizes function decorators, `@broker.subscriber` and `@broker.publisher` to make interacting with your brokers simple.

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

### Testing Your Service

Test your FastStream applications easily with the `TestBroker` context manager:

```python
import pytest
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1,}, "in")
```

### Running Your Application

Run your FastStream service using the built-in CLI:

```bash
pip install "faststream[cli]"
faststream run basic:app
```
Improve your development experience with hot-reloading and multiprocessing features:

```bash
faststream run basic:app --reload # Hot Reload
faststream run basic:app --workers 3 # Multiprocessing
```

## Project Documentation

FastStream automatically generates documentation in AsyncAPI format for easy service integration.
![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

## Advanced Usage

*   **Dependencies:** Manage dependencies using FastStream's dependency injection system.
*   **HTTP Framework Integrations:** Integrate FastStream with any HTTP framework, including seamless integration with FastAPI.

## Stay Connected

*   **GitHub:** Give us a star on our [GitHub repository](https://github.com/ag2ai/faststream/)
*   **Discord:** Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   **Telegram:** Join our [RU Telegram group](https://t.me/python_faststream)

## Contributors

A huge thank you to the amazing contributors who make FastStream great!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>