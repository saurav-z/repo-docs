# FastStream: Effortless Event Stream Integration for Modern Microservices

**Simplify your microservice architecture with FastStream, the Python framework that makes integrating event streams a breeze.**  [Explore FastStream on GitHub](https://github.com/ag2ai/faststream)

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream Shield](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   **Multiple Broker Support:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Easily validate and serialize messages using Pydantic for robust data handling.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation to automatically document your streams.
*   **Intuitive Development Experience:** Benefit from full-typed editor support, reducing errors and improving developer productivity.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Simplified Testing:** Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensible Architecture:** Customize behavior with extensions for lifespans, custom serialization, and middleware.
*   **Flexible Integrations:** Compatible with any HTTP framework, including a dedicated plugin for FastAPI.

---

**[Read the Full Documentation](https://faststream.ag2.ai/latest/)**

---

## History

FastStream builds upon the innovative ideas and best practices from FastKafka and Propan, offering a unified solution for processing streamed data across different protocols.  We continue to maintain these projects, but new development is focused here.

---

## Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-like systems. Install using pip:

```bash
pip install 'faststream[kafka]' # or [rabbit, nats, redis]
```

By default, FastStream uses PydanticV2 (written in Rust).  If Rust isn't supported on your platform, it will work with PydanticV1.

---

## Writing App Code

Use decorators like `@broker.subscriber` and `@broker.publisher` for easy message consumption and production. FastStream uses Pydantic for input message parsing.

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

Define messages with Pydantic's `BaseModel`:

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

## Testing the Service

Test your service using `TestBroker` with pytest.

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

---

## Running the Application

Install the CLI:

```bash
pip install "faststream[cli]"
```

Run your app:

```bash
faststream run basic:app
```

Use `--reload` for hot-reloading and `--workers` for multiprocessing.

---

## Project Documentation

FastStream automatically generates AsyncAPI documentation.

![AsyncAPI Documentation](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency system, similar to `pytest fixtures` and `FastAPI Depends`, allows you to manage dependencies using function arguments.

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

## HTTP Framework Integrations

### Any Framework

Use `MQBrokers` without a `FastStream` application by managing their lifecycles.

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

Integrate with FastAPI using `KafkaRouter`.

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

## Stay Connected

*   [GitHub Repository](https://github.com/ag2ai/faststream)
*   [EN Discord Server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram Group](https://t.me/python_faststream)

---

## Contributors

[Contributors Graph](https://github.com/ag2ai/faststream/graphs/contributors)