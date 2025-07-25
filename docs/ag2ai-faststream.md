# FastStream: Effortlessly Integrate Event Streams for Microservices

Tired of complex event stream integrations? **FastStream** simplifies building microservices with seamless message queue integration.  [Explore FastStream on GitHub](https://github.com/ag2ai/faststream/)

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

## Key Features of FastStream

*   **Multiple Broker Support:**  Integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Seamlessly serialize and validate messages using Pydantic for data integrity.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation for your service endpoints automatically.
*   **Intuitive Development Experience:** Benefit from full-typed editor support that catches errors early.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Simplified Testing:** Utilize in-memory tests for faster, more reliable CI/CD pipelines.
*   **Extensibility:** Use extensions for lifespans, custom serialization, and middleware.
*   **Framework-Agnostic Integration:** Compatible with any HTTP framework, including a dedicated [FastAPI plugin](#fastapi-plugin).

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## History

FastStream builds upon the strengths of FastKafka and Propan. It provides a unified approach to process streamed data, regardless of the underlying protocol.

---

## Installation

FastStream is compatible with various operating systems and is easily installed using pip:

```bash
pip install 'faststream[kafka]'  # for Kafka
# or
pip install 'faststream[rabbit]'  # for RabbitMQ
# or
pip install 'faststream[nats]'    # for NATS
# or
pip install 'faststream[redis]'   # for Redis
```

FastStream utilizes PydanticV2 by default, which is written in Rust. If your platform lacks Rust support, FastStream seamlessly integrates with PydanticV1.

---

## Writing App Code

FastStream simplifies message queue interactions via the `@broker.subscriber` and `@broker.publisher` decorators. These decorators handle tasks such as consuming and producing data, decoding, and encoding JSON-formatted messages. This allows you to concentrate on the core application logic.

Leverage Pydantic for data validation and serialization by specifying message types using type annotations.

Here's an example of a Python application utilizing FastStream:

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

You can also define message structures using Pydantic's `BaseModel` class:

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

Use `TestBroker` context managers to test your service. The `TestBroker` redirects the `subscriber` and `publisher` functions to in-memory brokers.

Here's how to test using pytest:

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

Use the FastStream CLI to run your application:

1.  Install the CLI:

```bash
pip install "faststream[cli]"
```

2.  Run your application:

```bash
faststream run basic:app
```

You can also use `--reload` for hot-reloading and `--workers` for multiprocessing:

```bash
faststream run basic:app --reload
faststream run basic:app --workers 3
```

Learn more about the CLI [here](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream automatically generates documentation for your project based on the AsyncAPI specification. This simplifies service integration by providing immediate insights into message formats and channels.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency management system similar to pytest fixtures and FastAPI Depends via [FastDepends](https://lancetnik.github.io/FastDepends/).

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

Use `MQBrokers` with any HTTP framework by starting and stopping them as part of your application's lifecycle.

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

Integrate FastStream with FastAPI. Import the `StreamRouter` and use the `@router.subscriber` and `@router.publisher` decorators.

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

More on FastAPI integration [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/).

---

## Stay Connected

Show your support and stay updated:

*   Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Thank you to all contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>