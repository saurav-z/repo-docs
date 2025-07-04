# FastStream: Effortlessly Integrate Event Streams for Microservices

**Build robust and scalable microservices with ease using FastStream, a Python framework designed to simplify event stream integration.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
<br/>
<br/>
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
<br/>
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
<br/>
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
<img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
<br/>
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support:** Seamlessly integrate with various message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization of incoming messages.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically to streamline service integration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in development.
*   **Dependency Injection:** Manage dependencies efficiently with FastStream's built-in dependency injection system.
*   **Testability:** Easily test your services with in-memory tests, improving CI/CD pipelines.
*   **Extensibility:** Extend functionality using extensions for lifespans, custom serialization, and middleware.
*   **Framework Integration:** Compatible with any HTTP framework, particularly [FastAPI](#fastapi-plugin).

---

**[Get Started with FastStream](https://faststream.airt.ai/latest/)**

---

## History

FastStream is a new project built upon the foundation of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), combining the best features of each to provide a unified approach to processing streamed data.

---

## Installation

FastStream supports Linux, macOS, Windows, and most Unix-style operating systems.

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

## Writing Application Code

Use the `@broker.subscriber` and `@broker.publisher` decorators to define message consumers and producers:

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

Leverage Pydantic for message schema validation:

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

Use `TestBroker` for in-memory testing:

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

Run your service:

```bash
faststream run basic:app
```

Use hot reload and multiprocessing:

```bash
faststream run basic:app --reload
faststream run basic:app --workers 3
```

---

## Project Documentation

FastStream automatically generates documentation compliant with the [AsyncAPI](https://www.asyncapi.com/) specification.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency management system similar to `pytest fixtures` and `FastAPI Depends`.

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

Use `MQBrokers` without a `FastStream` application:

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

Integrate with FastAPI using `KafkaRouter`:

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

*   Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Thanks to the contributors:

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>