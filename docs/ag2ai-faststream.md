# FastStream: Effortlessly Integrate Event Streams for Microservices 🚀

**FastStream empowers developers to build robust, data-driven microservices with ease.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Tests Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Package version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram (RU)](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support**: Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation**:  Leverage Pydantic for efficient data validation and serialization.
*   **Automatic Documentation**: Generate comprehensive AsyncAPI documentation automatically, simplifying service integration.
*   **Intuitive Development**: Benefit from full-typed editor support, catching errors early in the development process.
*   **Powerful Dependency Injection**:  Manage service dependencies effectively with FastStream's built-in DI system.
*   **Testability**:  Utilize in-memory tests to create faster and more reliable CI/CD pipelines.
*   **Extensibility**:  Customize your streaming applications with extensions for lifespans, serialization, and middleware.
*   **Framework Integrations**:  Works seamlessly with any HTTP framework, especially FastAPI.

---

## What is FastStream?

FastStream is a Python framework designed to streamline the development of event-driven microservices. It simplifies message queue interactions, handling parsing, networking, and documentation generation automatically. Built with developers in mind, FastStream helps you build modern, data-centric microservices with ease.

## How It Works

FastStream utilizes function decorators like `@broker.subscriber` and `@broker.publisher` to simplify the process of consuming and producing data. This allows you to focus on your core business logic without getting bogged down in the complexities of message queue integration. FastStream also uses Pydantic for data validation.

## Getting Started

### Installation

Install FastStream with your preferred broker:

```bash
pip install 'faststream[kafka]' # or [rabbit], [nats], [redis]
```

## Core Concepts

### Writing App Code

Use `@broker.subscriber` and `@broker.publisher` decorators to define message handlers, then specify how to process incoming messages.

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

### Testing the Service

Test your service using `TestKafkaBroker` context managers without needing a running broker:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1,}, "in")

@pytest.mark.asyncio
async def test_invalid():
    async with TestKafkaBroker(broker) as br:
        with pytest.raises(pydantic.ValidationError):
            await br.publish("wrong message", "in")
```

### Running the Application

Run your service with the FastStream CLI:

```bash
pip install "faststream[cli]"
faststream run basic:app
```

Use the `--reload` and `--workers` options for enhanced development and scaling.

## Integrations

FastStream integrates seamlessly with various frameworks.

### Any Framework

Integrate FastStream `MQBrokers` independently, starting and stopping them with your application's lifecycle.

### FastAPI Plugin

Easily incorporate FastStream into your FastAPI applications using `KafkaRouter`.

## Further Reading

*   **Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)
*   **Example:** [Basic Example](https://faststream.ag2.ai/latest/getting-started/basic_example/)
*   **CLI Documentation:** [FastStream CLI Features](https://faststream.ag2.ai/latest/getting-started/cli/)
*   **FastAPI Plugin:** [FastAPI Integration](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## History

FastStream is an evolution of FastKafka and Propan, bringing together the best features of both projects.

---

## Community

Stay connected and get support:

*   **GitHub Repository:** [ag2ai/faststream](https://github.com/ag2ai/faststream/)
*   **Discord:** [EN Discord server](https://discord.gg/qFm6aSqq59)
*   **Telegram:** [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

[<img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>](https://github.com/ag2ai/faststream/graphs/contributors)