# FastStream: Effortlessly Integrate Event Streams in Your Microservices

**Build scalable and efficient microservices with ease.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

FastStream is a Python framework designed to simplify the development of event-driven microservices by automating message queue interactions and providing comprehensive features for efficient data processing.

*   **Multiple Broker Support:** Integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Validate incoming messages using Pydantic's robust data validation capabilities.
*   **Automatic AsyncAPI Documentation:** Generate AsyncAPI documentation automatically for easy service integration and discoverability.
*   **Type-Safe Development:** Benefit from full-typed editor support for a smoother development experience and early error detection.
*   **Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Testability:** Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Extend functionality with extensions for lifespans, custom serialization, and middleware.
*   **Framework Agnostic:** Integrate seamlessly with any HTTP framework, including FastAPI.

---

## Documentation

*   [FastStream Documentation](https://faststream.ag2.ai/latest/)
*   [FastStream GitHub Repository](https://github.com/ag2ai/faststream)

---

## Getting Started

### Installation

Install FastStream with your desired broker support using pip:

```bash
pip install 'faststream[kafka]'
pip install 'faststream[rabbit]'
pip install 'faststream[nats]'
pip install 'faststream[redis]'
```

### Quick Start

Here's a basic example of how to use FastStream with Kafka:

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

### Testing

Use `TestKafkaBroker` for in-memory testing:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({
            "user": "John",
            "user_id": 1,
        }, "in")

@pytest.mark.asyncio
async def test_invalid():
    async with TestKafkaBroker(broker) as br:
        with pytest.raises(pydantic.ValidationError):
            await br.publish("wrong message", "in")
```

### Running the Application

1.  Install FastStream CLI: `pip install "faststream[cli]"`
2.  Run your application: `faststream run basic:app`
3.  For hot reload: `faststream run basic:app --reload`
4.  For multiprocessing: `faststream run basic:app --workers 3`

---

## Core Concepts

*   **Message Handling:** Use `@broker.subscriber` and `@broker.publisher` decorators to easily define message producers and consumers.
*   **Pydantic Integration:** Leverage Pydantic for data validation and serialization using type annotations.
*   **Project Documentation:** Automatically generate AsyncAPI documentation.
*   **Dependency Injection:** Use FastStream's built-in dependency injection system.

---

## Advanced Features and Integrations

### Dependency Injection

Manage dependencies with a system similar to `pytest fixtures` and `FastAPI Depends`:

```python
from typing import Annotated
from faststream import Depends, Logger

async def base_dep(user_id: int) -> bool:
    return True

@broker.subscriber("in-test")
async def base_handler(user: str,
                       logger: Logger,
                       dep: Annotated[bool, Depends(base_dep)]):
    assert dep is True
    logger.info(user)
```

### Framework Integrations

#### Any Framework

Integrate `MQBrokers` without a `FastStream` application using *start* and *stop* methods during your application's lifespan:

```python
from aiohttp import web
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")

@broker.subscriber("test")
async def base_handler(body):
    print(body)

async def start_broker(app):
    await broker.start()

async def stop_broker(app):
    await broker.stop()

async def hello(request):
    return web.Response(text="Hello, world")

app = web.Application()
app.add_routes([web.get("/", hello)])
app.on_startup.append(start_broker)
app.on_cleanup.append(stop_broker)

if __name__ == "__main__":
    web.run_app(app)
```

#### FastAPI Plugin

Integrate FastStream with FastAPI using `KafkaRouter` and decorators:

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

---

## Community and Support

*   [GitHub Repository](https://github.com/ag2ai/faststream)
*   [EN Discord Server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram Group](https://t.me/python_faststream)

---

## Contributors

[List of Contributors](https://github.com/ag2ai/faststream/graphs/contributors)

---