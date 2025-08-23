# FastStream: Effortlessly Build Modern, Data-Driven Microservices

**Simplify event stream integration and accelerate your microservice development with FastStream, the Python framework built for speed and ease.**  ([Original Repository](https://github.com/ag2ai/faststream))

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Package Version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord EN](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream Status](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multi-Broker Support:**  Seamlessly integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:**  Ensure data integrity with robust Pydantic validation for incoming messages.
*   **Automatic AsyncAPI Documentation:**  Generate comprehensive AsyncAPI documentation automatically, simplifying service integration and understanding.
*   **Intuitive Development Experience:** Benefit from full-typed editor support, catching errors early in development.
*   **Powerful Dependency Injection:**  Efficiently manage service dependencies with FastStream's built-in DI system.
*   **Simplified Testing:**  Leverage in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:**  Customize your services with extensions for lifespans, custom serialization, and middleware.
*   **Framework Compatibility:**  Integrate with any HTTP framework, with excellent support for FastAPI.

---

## Documentation
Access the full documentation here: [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## What is FastStream?

FastStream is built on the experience and expertise of FastKafka and Propan. It offers a unified approach to building services that process streamed data, providing a consistent experience across different messaging protocols. This makes it an excellent choice for new microservice projects.

---

## Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-like operating systems. Install it easily with pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream defaults to PydanticV2. If needed, you can downgrade to PydanticV1.

---

## Writing App Code

FastStream offers convenient function decorators (`@broker.subscriber` and `@broker.publisher`) to handle message queue interactions, including data serialization and deserialization. This allows you to focus on your core business logic.

FastStream uses Pydantic for seamless data parsing, allowing you to use type annotations to serialize input messages.

**Example:**

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

You can also define messages using Pydantic's `BaseModel`:

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

FastStream provides `TestBroker` context managers for simplified testing without requiring a running broker.

**Example (pytest):**

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

Run your application using the FastStream CLI. First, install the CLI:

```bash
pip install "faststream[cli]"
```

Then, run your service with:

```bash
faststream run basic:app
```

Additional features:

*   Hot reload: `faststream run basic:app --reload`
*   Multiprocessing: `faststream run basic:app --workers 3`

Learn more about the CLI [here](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream automatically generates [AsyncAPI](https://www.asyncapi.com/) documentation. This simplifies service integration by providing immediate visibility into channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency management system, built on [FastDepends](https://lancetnik.github.io/FastDepends/), supports features similar to `pytest fixtures` and `FastAPI Depends`.

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

Learn more about FastAPI integration [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/).

---

## Stay Connected

Show your support and stay updated by:

*   Starring our [GitHub repository](https://github.com/ag2ai/faststream/).
*   Joining our [Discord server](https://discord.gg/qFm6aSqq59).
*   Joining our [Telegram group](https://t.me/python_faststream).

---

## Contributors

Special thanks to the contributors who made this project possible!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>