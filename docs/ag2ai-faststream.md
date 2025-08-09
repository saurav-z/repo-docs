# FastStream: Effortlessly Integrate Event Streams into Your Microservices

**Build robust, data-driven microservices with ease using FastStream, a Python framework designed for efficient event stream integration.**

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

## Key Features of FastStream

*   **Multiple Broker Support:** Seamlessly integrate with [Kafka](https://kafka.apache.org/), [RabbitMQ](https://www.rabbitmq.com/), [NATS](https://nats.io/), and [Redis](https://redis.io/) using a unified API.
*   **Pydantic Validation:** Leverage [Pydantic's](https://docs.pydantic.dev/) robust validation for efficient data parsing and error handling.
*   **Automatic Documentation:** Generate comprehensive [AsyncAPI](https://www.asyncapi.com/) documentation automatically, simplifying service integration.
*   **Type-Safe Development:** Benefit from full-typed editor support for a smoother development experience and reduced runtime errors.
*   **Dependency Injection:** Utilize FastStream's built-in dependency injection system for efficient service management.
*   **Simplified Testing:** Quickly test your code with in-memory tests, improving your CI/CD pipeline speed.
*   **Extensibility:** Use extensions for lifespans, custom serialization and middleware
*   **Framework Integrations:**  Compatible with any HTTP framework and especially [FastAPI](#fastapi-plugin)

**Ready to get started?** Head over to the [FastStream GitHub Repository](https://github.com/ag2ai/faststream/) for the source code and more details.

---

## History

FastStream evolves from the successful experiences of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), unifying the best features to streamline event streaming. It is recommended for new services.

---

## Installation

FastStream is compatible with **Linux**, **macOS**, **Windows**, and most **Unix**-style systems.

Install with pip:

```bash
pip install 'faststream[kafka]'  # or
pip install 'faststream[rabbit]'  # or
pip install 'faststream[nats]'    # or
pip install 'faststream[redis]'
```

FastStream uses **PydanticV2** written in **Rust**, but you can downgrade to **PydanticV1** if needed.

---

## Writing App Code

FastStream uses function decorators `@broker.subscriber` and `@broker.publisher` to simplify producing and consuming data to and from event queues.

Also, FastStream uses [**Pydantic**](https://docs.pydantic.dev/) to parse input
JSON-encoded data into Python objects, making it easy to work with structured data in your applications, so you can serialize your input messages just using type annotations.

Example:

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

Use `TestBroker` for in-memory testing with pytest:

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

Hot reload:

```bash
faststream run basic:app --reload
```

Multiprocessing:

```bash
faststream run basic:app --workers 3
```

More CLI info [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates AsyncAPI documentation.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

Use a dependency management system like `pytest fixtures` and `FastAPI Depends`:

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

More integration features [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay in Touch

*   [GitHub Repository](https://github.com/ag2ai/faststream/)
*   [EN Discord server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>