# FastStream: Effortlessly Build and Scale Event-Driven Microservices

**Supercharge your microservices with FastStream – the Python framework for seamless event stream integration.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Tests Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage Status](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Monthly Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Package Version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)

---

## Key Features

*   **Simplified Event Streaming:**  Easily build producers and consumers with intuitive decorators, automatically handling parsing, networking, and documentation.
*   **Multiple Broker Support:**  Integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization of incoming messages.
*   **Automatic AsyncAPI Documentation:** Generate up-to-date AsyncAPI documentation to simplify service integration.
*   **Type-Safe Development:**  Benefit from full-typed editor support, catching errors early in development.
*   **Dependency Injection:**  Manage service dependencies efficiently with a built-in dependency injection system.
*   **Testability:**  Use in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:**  Customize behavior with extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:**  Seamlessly integrate with any HTTP framework, especially FastAPI.

---

**Get started:**  [FastStream Documentation](https://faststream.ag2.ai/latest/)

---

## What is FastStream?

FastStream is a powerful Python framework designed to simplify building and scaling event-driven microservices. It streamlines the complexities of message queue integration, offering a user-friendly experience for both novice and experienced developers.

## History

FastStream is built upon the successful concepts of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), combining the best features of both to offer a unified approach to handling streamed data.  New development focuses on this project.

---

## Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-style operating systems.

```bash
pip install 'faststream[kafka]'  # For Kafka
# or
pip install 'faststream[rabbit]' # For RabbitMQ
# or
pip install 'faststream[nats]'   # For NATS
# or
pip install 'faststream[redis]'  # For Redis
```

FastStream defaults to PydanticV2, but supports PydanticV1 if required by your platform.

---

## Writing App Code

FastStream provides convenient function decorators `@broker.subscriber` and `@broker.publisher` to simplify your message handling logic.

Key aspects:
*   Consuming and producing data
*   Decoding and encoding JSON-encoded messages

Here's a basic example:

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

Using Pydantic models:

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

FastStream's `TestBroker` context managers simplify testing.

Here's a pytest example:

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

Install the FastStream CLI:

```bash
pip install "faststream[cli]"
```

Run your app:

```bash
faststream run basic:app
```

Features:

*   Hot reload: `faststream run basic:app --reload`
*   Multiprocessing: `faststream run basic:app --workers 3`
*   Learn more about FastStream CLI: [CLI Documentation](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream auto-generates documentation in [AsyncAPI](https://www.asyncapi.com/) format.

![AsyncAPI HTML page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency injection system (built on [FastDepends](https://lancetnik.github.io/FastDepends/)) works similarly to `pytest fixtures` and `FastAPI Depends`.

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

Integrate `MQBrokers` without a `FastStream` app:

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

Integrate FastStream with FastAPI.

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

More integration features: [FastAPI integration](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

*   Star the [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

[Contributors Image](https://contrib.rocks/image?repo=ag2ai/faststream)

---

**[Go back to the FastStream repository.](https://github.com/ag2ai/faststream)**