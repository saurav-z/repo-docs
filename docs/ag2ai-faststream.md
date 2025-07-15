# FastStream: Effortlessly Integrate Event Streams for Modern Microservices

**Simplify your microservices architecture with FastStream, a Python framework for seamless event stream integration.**

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
[![FastStream Shield](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   **Multiple Brokers:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization.
*   **Automatic Docs:** Generate AsyncAPI documentation automatically for easy service understanding.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in development.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with a built-in DI system.
*   **Testable:** Simplify testing with in-memory tests for faster CI/CD cycles.
*   **Extensible:** Leverage extensions for lifespans, custom serialization, and middleware.
*   **Integrations:** Integrate with any HTTP framework, especially FastAPI.

---

**For more details, visit the documentation: [FastStream Documentation](https://faststream.ag2.ai/latest/)**

---

## What is FastStream?

FastStream is a Python framework designed to simplify the development of streaming microservices. It provides a streamlined approach to writing producers and consumers for message queues, handling parsing, networking, and documentation generation automatically. Built with developer experience in mind, FastStream simplifies complex tasks while allowing for advanced customization, making it a go-to solution for modern, data-centric microservices.

---

## History

FastStream builds upon the success of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), integrating the best features from both projects into a unified way to handle streamed data processing.  If you're starting a new streaming service, FastStream is the recommended approach.

---

## Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-style operating systems.

Install using pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream defaults to using PydanticV2 (written in Rust), but if your platform lacks Rust support, it will automatically work with PydanticV1.

---

## Writing App Code

FastStream simplifies event stream integration using decorators like `@broker.subscriber` and `@broker.publisher`.

These decorators handle consuming, producing, and encoding/decoding JSON messages, simplifying your business logic.

Leverage [Pydantic](https://docs.pydantic.dev/) for data validation by simply defining message types using type annotations:

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

You can also define your messages using Pydantic's `BaseModel`:

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

Test your service efficiently with `TestBroker` context managers:

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

Run your application using the FastStream CLI.

Install the CLI:

```bash
pip install "faststream[cli]"
```

Run your app:

```bash
faststream run basic:app
```

Improve development with hot reload:

```bash
faststream run basic:app --reload
```

And enable multiprocessing for horizontal scaling:

```bash
faststream run basic:app --workers 3
```

Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates project documentation in the [AsyncAPI](https://www.asyncapi.com/) specification.  Easily understand your service's channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency management uses a system similar to `pytest fixtures` and `FastAPI Depends`:

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

## HTTP Frameworks Integrations

### Any Framework

Integrate `MQBrokers` without a `FastStream` application:

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

Use FastStream with FastAPI:

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

Find more integration features [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay in Touch

Support FastStream by:

*   Starring our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Joining our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Joining our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thank you to all contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>