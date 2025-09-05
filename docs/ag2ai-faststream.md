# FastStream: Effortlessly Integrate Event Streams for Modern Microservices

**Simplify your event-driven architectures with FastStream, the Python framework designed for seamless integration with message brokers.**

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
[![FastStream Shield](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram (RU)](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

FastStream provides a streamlined experience for building event-driven microservices.

*   **Multi-Broker Support:** Work seamlessly with Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Validate your messages with Pydantic for robust data handling.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early.
*   **Dependency Injection:** Manage service dependencies efficiently with built-in DI system.
*   **Simplified Testing:** Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensible Architecture:** Leverage extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:** Compatible with any HTTP framework, especially [FastAPI](#fastapi-plugin).

---

**[Get Started with FastStream](https://faststream.ag2.ai/latest/)**

---

## Why Choose FastStream?

FastStream simplifies the complexities of event streaming, making it easier than ever to build robust, scalable, and maintainable microservices.  It is designed to be accessible to developers of all skill levels, providing a powerful yet intuitive framework.

---

## Installation

Install FastStream using pip:

```bash
pip install 'faststream[kafka]' # Or other brokers like 'rabbit', 'nats', or 'redis'
```

---

## Basic Usage Example

Define your event-driven logic with ease using decorators and type hints:

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

You can also use Pydantic models for message definition and validation:

```python
from pydantic import BaseModel, Field, PositiveInt
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

class User(BaseModel):
    user: str = Field(..., examples=["John"])
    user_id: PositiveInt = Field(..., examples=["1"])

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(data: User) -> str:
    return f"User: {data.user} - {data.user_id} registered"
```

---

## Testing Your Service

FastStream simplifies testing with `TestBroker`:

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
```

---

## Running Your Application

Run your FastStream application using the CLI:

```bash
pip install "faststream[cli]"
faststream run basic:app
```

Enhance your development workflow with hot reload and multiprocessing:

```bash
faststream run basic:app --reload
faststream run basic:app --workers 3
```

---

## Project Documentation

FastStream automatically generates [AsyncAPI](https://www.asyncapi.com/) documentation for your project.  This documentation simplifies integration and provides an overview of your service's channels and message formats.

---

## Dependencies

Leverage a dependency management system similar to `pytest fixtures` and `FastAPI Depends`:

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

---

## Framework Integrations

### Any Framework

Integrate FastStream's `MQBrokers` with any framework:

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

### FastAPI Plugin

Easily integrate FastStream with [FastAPI](https://fastapi.tiangolo.com/):

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

## Get Involved

*   ⭐ Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   💬 Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   🗣️ Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Thank you to the amazing contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>