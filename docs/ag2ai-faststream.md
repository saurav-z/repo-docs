# FastStream: Effortlessly Integrate Event Streams into Your Microservices 🚀

**Streamline your microservice communication with FastStream, a powerful framework that simplifies event-driven architecture.** (Link back to original repo: [https://github.com/ag2ai/faststream](https://github.com/ag2ai/faststream))

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

*   **Multi-Broker Support:** Seamlessly work with various message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization of your message payloads.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically, making it easy to understand your service's API.
*   **Intuitive Development Experience:** Enjoy full-typed editor support, catching errors early and improving developer productivity.
*   **Powerful Dependency Injection:** Easily manage dependencies within your service using FastStream's built-in DI system.
*   **Testable with Ease:** Leverage in-memory tests for fast and reliable CI/CD pipelines.
*   **Extensible:** Customize your service with lifespans, custom serialization, and middleware extensions.
*   **Framework Integrations:** Fully compatible with any HTTP framework (especially FastAPI).

---

## Installation

Install FastStream using pip:

```bash
pip install 'faststream[kafka]'  # or 'faststream[rabbit]', 'faststream[nats]', 'faststream[redis]'
```

FastStream uses PydanticV2 written in Rust but will work with PydanticV1 if the platform has no Rust support.

---

## Writing Application Code

FastStream provides convenient decorators (`@broker.subscriber` and `@broker.publisher`) to simplify message handling:

*   Consume and Produce data with Event queues
*   Decode and Encode JSON-encoded messages

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

**Pydantic Integration:**
Define messages with a declarative syntax.

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

Test your service with `TestBroker` context managers.

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

Install the FastStream CLI:

```bash
pip install "faststream[cli]"
```

Run your application:

```bash
faststream run basic:app
```

Features:
- Hot reload: `faststream run basic:app --reload`
- Multiprocessing: `faststream run basic:app --workers 3`

---

## Project Documentation

FastStream automatically generates AsyncAPI documentation, simplifying service integration.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

Use dependency injection via function arguments with the `Depends` decorator.

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

*   [GitHub Repository](https://github.com/ag2ai/faststream/)
*   [EN Discord Server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram Group](https://t.me/python_faststream)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>