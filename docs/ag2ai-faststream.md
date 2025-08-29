# FastStream: Simplify Your Microservices with Effortless Event Stream Integration

**Quickly build robust and scalable microservices with FastStream, the Python framework for effortless event stream integration.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
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

FastStream provides a streamlined approach to build and manage event-driven microservices. It simplifies the complexities of message queue interactions, allowing developers to focus on core business logic. Here's what makes FastStream stand out:

*   **Multiple Broker Support**: Seamlessly integrate with various message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation**: Validate incoming messages using Pydantic's robust validation capabilities to ensure data integrity.
*   **Automatic AsyncAPI Documentation**: Generate comprehensive AsyncAPI documentation automatically, simplifying service integration and discovery.
*   **Intuitive Development Experience**: Benefit from full-typed editor support, which identifies errors early and enhances development speed.
*   **Powerful Dependency Injection**: Leverage FastStream's built-in dependency injection system for efficient service management.
*   **Simplified Testing**: Utilize in-memory testing capabilities to create faster and more reliable CI/CD pipelines.
*   **Extensibility**: Customize lifespans, serialization, and middleware with ease.
*   **Framework Compatibility**: Integrate FastStream seamlessly with any HTTP framework, with specific support for FastAPI.

For more details, check out the [official FastStream documentation](https://faststream.ag2.ai/latest/).

---

## Getting Started

### Installation

FastStream is designed for Linux, macOS, Windows, and most Unix-style operating systems. Install it using pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

### Basic Usage

FastStream makes it easy to write producers and consumers, abstracting away the complexities of message queue interactions. Use decorators like `@broker.subscriber` and `@broker.publisher` to streamline data processing and focus on core business logic. Utilize Pydantic for data serialization and validation.

Here’s a simple example:

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

Or with Pydantic Models:

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

### Testing

Test your service with `TestKafkaBroker`:

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

Install the FastStream CLI and then run your app:

```bash
pip install "faststream[cli]"
faststream run basic:app
```

Enhance your development experience with hot reload and multiprocessing:

```bash
faststream run basic:app --reload
faststream run basic:app --workers 3
```

### Project Documentation

FastStream automatically generates AsyncAPI documentation for your projects. This documentation simplifies the integration of services by clearly defining channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

### Dependencies

FastStream's dependency management system is inspired by pytest fixtures and FastAPI's dependency system.

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

Integrate `MQBrokers` with any framework:

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

Easily integrate with FastAPI:

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

## Community & Support

*   ⭐ Star our [GitHub repository](https://github.com/ag2ai/faststream/) to show your support.
*   💬 Join our [EN Discord server](https://discord.gg/qFm6aSqq59) for discussions and support.
*   🇷🇺 Join our [RU Telegram group](https://t.me/python_faststream) for Russian-speaking users.

---

## Contribute

We welcome contributions!  Please see the [original repo](https://github.com/ag2ai/faststream/) for guidelines.

---

## Contributors

A huge thank you to all contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>