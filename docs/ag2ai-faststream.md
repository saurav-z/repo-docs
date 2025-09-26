# FastStream: Effortlessly Integrate Event Streams for Microservices

**Streamline your microservices with FastStream, the Python framework for effortless event stream integration.**  [Check out the original repo](https://github.com/ag2ai/faststream).

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Tests Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multi-Broker Support:** Integrate with various message brokers, including [Kafka](https://kafka.apache.org/), [RabbitMQ](https://www.rabbitmq.com/), [NATS](https://nats.io/), and [Redis](https://redis.io/).
*   **Pydantic Validation:** Utilize [Pydantic](https://docs.pydantic.dev/) for robust data validation, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation for your streaming APIs, simplifying service integration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in the development process.
*   **Dependency Injection:** Leverage a built-in dependency injection system for efficient service management.
*   **Testability:**  Use in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Customize your application with extensions for lifespans, custom serialization, and middleware.
*   **Framework Compatibility:** Seamlessly integrate with any HTTP framework, particularly [FastAPI](https://fastapi.tiangolo.com/).

---

## Documentation

Explore the comprehensive [FastStream documentation](https://faststream.ag2.ai/latest/) to learn more.

---

## History

FastStream builds upon the successes of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), uniting the best features for modern, data-centric microservices.

---

## Installation

Install FastStream using `pip`:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

---

## Getting Started

FastStream simplifies event stream integration with function decorators and Pydantic.

### Example: Consuming and Producing Data

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

### Example: Data Validation with Pydantic

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

## Testing

Easily test your service using the `TestBroker` context manager.

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

Run your FastStream application using the CLI:

1.  Install the CLI:  `pip install "faststream[cli]"`
2.  Run your app:  `faststream run basic:app`

Enhance your development with hot reload (`--reload`) and multiprocessing (`--workers <number>`).

---

## Project Documentation

FastStream generates [AsyncAPI](https://www.asyncapi.com/) documentation automatically to visualize your streaming APIs and facilitate integration.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses [FastDepends](https://lancetnik.github.io/FastDepends/) for a dependency injection system.

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

Integrate `MQBrokers` without a `FastStream` application by managing their lifecycles.

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

Easily integrate with [FastAPI](https://fastapi.tiangolo.com/) using the `KafkaRouter`.

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

*   [GitHub Repository](https://github.com/ag2ai/faststream/) - Star the project!
*   [Discord Server](https://discord.gg/qFm6aSqq59) - Join the English-speaking community!
*   [Telegram Group](https://t.me/python_faststream) - Join the Russian-speaking community!

---

## Contributors

A big thank you to all contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>