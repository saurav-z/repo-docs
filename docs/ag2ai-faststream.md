# FastStream: Effortlessly Build Event-Driven Microservices in Python

**Simplify your event streaming with FastStream, a Python framework that accelerates microservice development.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
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

*   **Multiple Broker Support:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Validate messages using Pydantic for data integrity.
*   **Automatic AsyncAPI Docs:** Generate comprehensive documentation to streamline service integration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early.
*   **Dependency Injection:** Utilize a powerful DI system for efficient service management.
*   **Testable Architecture:** Built-in support for in-memory testing, making CI/CD faster.
*   **Extensible:** Leverage extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:** Compatible with any HTTP framework, with a dedicated FastAPI plugin.

---

## What is FastStream?

FastStream is a modern Python framework designed to simplify building and deploying event-driven microservices.  It provides a streamlined approach to working with message queues, reducing boilerplate and enabling developers to focus on business logic.  Built with a focus on ease of use and advanced features, FastStream is a great choice for both newcomers and experienced developers looking to build scalable and reliable microservices.  Learn more about FastStream at [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/).

---

## Getting Started

### Installation

Install FastStream using pip, selecting your desired broker:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream uses PydanticV2 by default, but if you encounter compatibility issues, you can downgrade manually.

### Example Code

Quickly build producers and consumers with decorators to automatically handle parsing, networking, and documentation generation.

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

Use `TestKafkaBroker` or the appropriate broker test class for your broker to simulate and validate your application in a test environment.

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

Use the FastStream CLI to run your application:

```bash
pip install "faststream[cli]"
faststream run basic:app
```

For hot-reloading: `faststream run basic:app --reload`

For multiprocessing: `faststream run basic:app --workers 3`

---

## Documentation

Detailed documentation is available at [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/).

---

## History

FastStream evolves from FastKafka and Propan, combining the best features of both to streamline the creation of services that process streaming data.

---

## Project Documentation

FastStream automatically generates AsyncAPI documentation.  This simplifies service integration by clearly defining channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

Manage dependencies like `pytest fixtures` and `FastAPI Depends` with FastStream's system.

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

## HTTP Frameworks integrations

### Any Framework

Use MQBrokers without a FastStream application.

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

### **FastAPI** Plugin

Integrate FastStream within FastAPI using a `StreamRouter`.

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

## Connect With Us

Show your support and stay up-to-date:

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thanks to all of these amazing people who made the project better!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>