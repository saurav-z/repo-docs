# FastStream: Effortlessly Build Modern Microservices with Event Streams

**Simplify your event-driven architecture and accelerate development with FastStream, a Python framework for building robust and scalable microservices.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream Shield](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram (RU)](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

FastStream makes building streaming microservices easy and efficient, even for junior developers. Here’s why it’s a great choice for modern, data-centric microservices:

*   ✅ **Multiple Broker Support:** Integrate seamlessly with Kafka, RabbitMQ, NATS, and Redis.
*   ✅ **Pydantic Validation:** Leverage Pydantic for robust data validation and serialization.
*   ✅ **Automatic AsyncAPI Documentation:** Automatically generate up-to-date documentation for your services.
*   ✅ **Developer-Friendly:** Benefit from intuitive, fully-typed editor support to catch errors early.
*   ✅ **Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   ✅ **Simplified Testing:** Easily test your services with in-memory tests.
*   ✅ **Extensible:** Use extensions for lifespans, custom serialization and middleware.
*   ✅ **Framework Integrations:** Seamlessly integrate with any HTTP framework, especially FastAPI.

For more details on the capabilities, check out the [FastStream Documentation](https://faststream.ag2.ai/latest/)

---

## Core Concepts

### **What is FastStream?**

FastStream is a modern Python framework designed to streamline the development of event-driven microservices. It simplifies the complexities of integrating with message queues, providing a unified API and automating tasks like parsing, networking, and documentation generation. Built with junior developers in mind, FastStream makes streaming microservices easier.

### **Why Use FastStream?**

FastStream aims to simplify the process of building streaming microservices, handling the complexities of message queue integration automatically. Key benefits include:

*   **Faster Development:** Reduce boilerplate and focus on business logic.
*   **Improved Code Quality:** Benefit from Pydantic validation and type hints.
*   **Simplified Operations:** Automatic documentation generation and easy testing.
*   **Scalability:** Designed to support scalable, event-driven architectures.

### **Comparison with Existing Tools**

*   **FastStream vs. Existing Message Queue Libraries:** FastStream offers a higher-level abstraction, simplifying broker interactions and providing features like automatic documentation and dependency injection.
*   **FastStream vs. Other Frameworks:** FastStream is specifically designed for event-driven architectures, providing a focused feature set and tighter integration with brokers.

---

## Getting Started

### Installation

Install FastStream using pip:

```bash
pip install "faststream[kafka]" # or rabbit, nats, redis
```

### Example Usage

Here's a basic example of using FastStream with Kafka:

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

### Key Features Explained

*   **Message Handling:** Use decorators like `@broker.subscriber` and `@broker.publisher` to define message consumers and producers.
*   **Pydantic Integration:** Easily validate and serialize messages with Pydantic models.
*   **Testing:** Use `TestKafkaBroker` to unit test your service.
*   **CLI:** Use the CLI to run and manage your FastStream application with commands like `faststream run <module>:<app>`.

---

## Advanced Topics

### Testing the Service

Use the `TestBroker` context managers to test your application logic without the need for a live broker.

```python
import pytest
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1}, "in")
```

### Running the Application

Run your FastStream application using the CLI:

```bash
faststream run basic:app
```

You can also use features like hot-reloading and multiprocessing.

### Project Documentation

FastStream generates AsyncAPI documentation automatically.

---

## Integrations

### FastAPI Plugin

Integrate FastStream seamlessly with FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from faststream.kafka.fastapi import KafkaRouter

router = KafkaRouter("localhost:9092")

class Incoming(BaseModel):
    m: dict

@router.subscriber("test")
@router.publisher("response")
async def hello(m: Incoming):
    return {"response": "Hello, world!"}

app = FastAPI()
app.include_router(router)
```

### Any Framework

You can use `MQBrokers` in any aiohttp application.

---

## Contributing

We welcome contributions! Find out how to get involved by visiting our [GitHub repository](https://github.com/ag2ai/faststream/).

---

## Community

*   **GitHub:** [https://github.com/ag2ai/faststream/](https://github.com/ag2ai/faststream/)
*   **Discord:** [https://discord.gg/qFm6aSqq59](https://discord.gg/qFm6aSqq59)
*   **Telegram (RU):** [https://t.me/python_faststream](https://t.me/python_faststream)

---

## License

[MIT License](https://github.com/ag2ai/faststream/blob/main/LICENSE)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>