# FastStream: Build Microservices with Ease Using Event Streaming

**Effortlessly integrate event streams into your services with FastStream, the Python framework that simplifies message queue interaction.**

[GitHub Repository](https://github.com/ag2ai/faststream)

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Tests](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Brokers**: Seamlessly work with various message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation**:  Use Pydantic for efficient data validation and serialization of incoming messages.
*   **Automatic Documentation**: Generate AsyncAPI documentation automatically to visualize your services.
*   **Intuitive Development**: Benefit from full-typed editor support, ensuring a smooth development experience and early error detection.
*   **Powerful Dependency Injection**: Manage service dependencies effectively with FastStream's built-in DI system.
*   **Testable**: Easily test your code with in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensible**: Leverage extensions for lifespans, custom serialization, and middleware.
*   **Framework Agnostic**: Integrate FastStream with any HTTP framework, with dedicated support for FastAPI.

FastStream empowers you to build modern, data-centric microservices with ease.

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## History

FastStream builds upon the insights from FastKafka and Propan, merging the best features to provide a unified approach to processing streamed data, regardless of the underlying protocol.  New development will focus on this project.

---

## Installation

FastStream supports Linux, macOS, Windows, and most Unix-style operating systems. Install it easily using pip:

```bash
pip install 'faststream[kafka]'  # For Kafka support
pip install 'faststream[rabbit]' # For RabbitMQ support
pip install 'faststream[nats]'   # For NATS support
pip install 'faststream[redis]'  # For Redis support
```

FastStream uses PydanticV2 written in Rust by default, but will work with PydanticV1 if Rust is not supported on your platform.

---

## Writing Application Code

FastStream simplifies the process of creating message producers and consumers, handling parsing, networking, and documentation generation automatically.

Use the `@broker.subscriber` and `@broker.publisher` decorators to define your message handling logic, focusing on your application's core functionality.  Also, leverage Pydantic for data serialization and validation via type annotations.

Here's an example of a FastStream app:

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker
# from faststream.rabbit import RabbitBroker
# from faststream.nats import NatsBroker
# from faststream.redis import RedisBroker

broker = KafkaBroker("localhost:9092")
# broker = RabbitBroker("amqp://guest:guest@localhost:5672/")
# broker = NatsBroker("nats://localhost:4222/")
# broker = RedisBroker("redis://localhost:6379/")

app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

Use Pydantic's `BaseModel` to define message structures:

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

Test your service using `TestBroker` context managers, which puts the Broker into "testing mode". This allows you to test your app without a running broker.

Example test with pytest:

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

The application can be started using built-in **FastStream CLI** command.

First, install the CLI:

```bash
pip install "faststream[cli]"
```

Then, run your service:

```bash
faststream run basic:app
```

Features available:

```bash
faststream run basic:app --reload
```
And multiprocessing horizontal scaling feature as well:
```bash
faststream run basic:app --workers 3
```
Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation for your project using the AsyncAPI specification. This simplifies service integration by providing clear insights into channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream leverages FastDepends for a robust dependency management system, similar to pytest fixtures and FastAPI Depends.

```python
from faststream import Depends, Logger

async def base_dep(user_id: int) -> bool:
    return True

@broker.subscriber("in-test")
async def base_handler(user: str,
                       logger: Logger,
                       dep: bool = Depends(base_dep)):
    assert dep is True
    logger.info(user)
```

---

## HTTP Framework Integrations

### Any Framework

Use FastStream `MQBrokers` independently from a `FastStream` application by *starting* and *stopping* them based on your application's lifecycle.

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

Integrate FastStream with FastAPI using `StreamRouter`. Declare message handlers with `@router.subscriber(...)` and `@router.publisher(...)` decorators.

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

More integration features can be found [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

Support FastStream by:

*   Starring our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Joining our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Joining our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thank you to all the contributors who make FastStream better!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>