# FastStream: Effortlessly Integrate Event Streams for Modern Microservices

**Simplify your microservices architecture with FastStream, the Python framework designed for seamless event stream integration.**  [Learn more about FastStream](https://faststream.ag2.ai/latest/)

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
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   **Multiple Broker Support:**  Integrate with a variety of message brokers, including Kafka, RabbitMQ, NATS, and Redis, using a unified API.
*   **Pydantic Validation:** Leverage Pydantic for robust data validation and serialization, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:**  Generate comprehensive documentation automatically, simplifying service integration and understanding.
*   **Intuitive Development Experience:**  Benefit from full-typed editor support, reducing errors and improving developer productivity.
*   **Dependency Injection:**  Manage service dependencies efficiently with FastStream's built-in DI system, promoting modularity and testability.
*   **Easy Testing:** Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Customize your implementation through extensions, including lifespans, custom serialization, and middleware.
*   **Framework Compatibility:**  Seamlessly integrates with any HTTP framework, with dedicated support for FastAPI.

---

**Explore the documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## Why FastStream? (History)

FastStream evolved from the successful projects FastKafka and Propan. This merger of strengths creates a powerful and unified solution for processing streaming data. If you're starting a new service, FastStream is the recommended approach.

---

## Installation

FastStream supports **Linux**, **macOS**, **Windows**, and most **Unix**-style operating systems.

Install FastStream using `pip`:

```bash
pip install 'faststream[kafka]'  # For Kafka support
# or
pip install 'faststream[rabbit]' # For RabbitMQ support
# or
pip install 'faststream[nats]'   # For NATS support
# or
pip install 'faststream[redis]'  # For Redis support
```

**Note:** FastStream uses PydanticV2 (Rust) by default, but it will work with PydanticV1 if your platform does not support Rust.

---

## Getting Started: Writing App Code

FastStream simplifies consuming and producing data to event queues using convenient decorators.

*   `@broker.subscriber`:  For consuming data.
*   `@broker.publisher`:  For publishing data.

These decorators handle the complexities of parsing, networking and documentation generation, letting you focus on your core logic.

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

You can use Pydantic models:

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

Test your service efficiently using `TestBroker` context managers for in-memory testing.

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

Use the FastStream CLI to run your application.

1.  **Install the CLI:** `pip install "faststream[cli]"`
2.  **Run your app:** `faststream run basic:app`

Enhance your development with:

*   Hot Reload: `faststream run basic:app --reload`
*   Multiprocessing: `faststream run basic:app --workers 3`

---

## Project Documentation & Benefits

FastStream automatically generates AsyncAPI documentation for your project, improving service integration and understanding.

![AsyncAPI Example](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses FastDepends for dependency management, similar to `pytest fixtures` and `FastAPI Depends`.

**Example:**

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

## Integrations

### Any Framework

Use FastStream's `MQBrokers` independently:

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

Integrate FastStream with FastAPI:

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

## Stay Connected & Support

*   ⭐ Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star!
*   💬 Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   🇷🇺 Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

[See the contributors](https://github.com/ag2ai/faststream/graphs/contributors)