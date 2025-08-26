# FastStream: Effortlessly Integrate Event Streams into Your Microservices

**Build robust, data-driven microservices faster with FastStream.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
<br/>
<br/>
[![Tests Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
<br/>
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
<br/>
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream Shield](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
<br/>
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multi-Broker Support:** Easily integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:**  Ensure data integrity with seamless integration with Pydantic for data validation and serialization.
*   **Automatic Documentation:** Generate and publish comprehensive AsyncAPI documentation for easy service understanding and integration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early for a smoother development process.
*   **Dependency Injection:** Manage your dependencies efficiently with a built-in, pytest-fixtures and FastAPI-depends-inspired DI system.
*   **Simplified Testing:**  Utilize in-memory testing to accelerate your CI/CD pipelines and ensure reliable deployments.
*   **Extensible Architecture:** Extend functionality through the use of extensions for lifespans, custom serialization, and middleware.
*   **Framework Agnostic:** Works seamlessly with any HTTP framework, offering dedicated integration with FastAPI.

**[Explore FastStream's documentation for more details.](https://faststream.ag2.ai/latest/)**

---

## Getting Started

### Installation

Install FastStream with your preferred broker:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream primarily relies on PydanticV2 for optimal performance; however, it seamlessly supports PydanticV1 for platforms lacking Rust support.

### Writing Application Code

FastStream uses decorators `@broker.subscriber` and `@broker.publisher` to simplify data consumption and production from event queues, eliminating manual parsing, networking, and documentation generation. This enables you to focus on core business logic.

Leverage Pydantic for type-safe data validation, easily serializing input messages using type annotations.

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

Use Pydantic’s [`BaseModel`](https://docs.pydantic.dev/usage/models/) for declarative message definition:

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

Use the `TestBroker` context manager for in-memory tests, streamlining your testing workflow.

**Example:**

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

## Running Your Application

Run your application with the FastStream CLI:

1.  **Install CLI:** `pip install "faststream[cli]"`
2.  **Run Service:** `faststream run basic:app`

Additional features include:

*   **Hot Reload:** `faststream run basic:app --reload`
*   **Multiprocessing:** `faststream run basic:app --workers 3`

[Learn more about FastStream CLI](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates AsyncAPI documentation, streamlining service integration.  Easily understand the channels and message formats your application uses.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency management system is similar to pytest fixtures and FastAPI Depends system.

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

Integrate `MQBrokers` from FastStream with any framework by managing their *start* and *stop* methods based on your application's lifecycle.

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

Integrate FastStream with FastAPI using the `StreamRouter`.

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
[More integration features.](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

*   [Star us on GitHub](https://github.com/ag2ai/faststream/)
*   [Join our Discord](https://discord.gg/qFm6aSqq59)
*   [Join our Telegram (RU)](https://t.me/python_faststream)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>