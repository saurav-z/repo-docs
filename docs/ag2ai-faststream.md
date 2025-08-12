# FastStream: Effortlessly Integrate Event Streams for Microservices

**Build robust and scalable microservices with ease.** FastStream simplifies event stream integration, empowering developers to focus on business logic, not infrastructure. [**Explore FastStream on GitHub**](https://github.com/ag2ai/faststream/).

---

<p align="center">
  <a href="https://trendshift.io/repositories/3043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3043" alt="ag2ai%2Ffaststream | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

  <br/>
  <br/>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main" alt="Test Passing"/>
  </a>

  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream" target="_blank">
      <img src="https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg" alt="Coverage"/>
  </a>

  <a href="https://www.pepy.tech/projects/faststream" target="_blank">
    <img src="https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/faststream" target="_blank">
    <img src="https://img.shields.io/pypi/v/faststream?label=PyPI" alt="Package version"/>
  </a>

  <a href="https://pypi.org/project/faststream" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/faststream.svg" alt="Supported Python versions"/>
  </a>

  <br/>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg" alt="CodeQL"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg" alt="Dependency Review"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/ag2ai/faststream.svg" alt="License"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md" target="_blank">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Code of Conduct"/>
  </a>

  <br/>

  <a href="https://discord.gg/qFm6aSqq59" target="_blank">
      <img alt="Discord" src="https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN"/>
  </a>

  <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>

  <a href="https://t.me/python_faststream" target="_blank">
    <img alt="Telegram" src="https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU"/>
  </a>

  <br/>

  <a href="https://gurubase.io/g/faststream" target="_blank">
    <img src="https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF" alt="Gurubase"/>
  </a>
</p>

---

## Key Features

*   **Unified Broker API:** Seamlessly work with multiple message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Easily validate and serialize messages using Pydantic for robust data handling.
*   **Automatic AsyncAPI Documentation:** Automatically generate API documentation, saving you time and effort.
*   **Type-Safe Development:** Benefit from full-typed editor support, catching errors early in the development process.
*   **Dependency Injection:** Utilize FastStream's built-in dependency injection system for efficient service management.
*   **Simplified Testing:** Leverage in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensible Architecture:** Utilize extensions for custom lifespans, serialization, and middleware.
*   **Framework Integrations:** Integrate with any HTTP framework, especially with the FastAPI plugin.

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## Installation

FastStream supports **Linux**, **macOS**, **Windows**, and most **Unix**-style operating systems. Install with pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream uses PydanticV2 written in Rust. If your platform doesn't support Rust, it will work correctly with PydanticV1.

---

## Getting Started

FastStream simplifies event stream integration with function decorators for publishers (`@broker.publisher`) and subscribers (`@broker.subscriber`), handling message parsing and encoding automatically.

### Basic Example

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

### Pydantic Integration

Use Pydantic models to define message structures for type-safe and validated data handling:

```python
from pydantic import BaseModel, Field
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

class User(BaseModel):
    user: str = Field(..., examples=["John"])
    user_id: int = Field(..., examples=[1])

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(data: User) -> str:
    return f"User: {data.user} - {data.user_id} registered"
```

---

## Testing

Use `TestKafkaBroker` (or the appropriate broker test class) with pytest for in-memory testing:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1}, "in")

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

Run your application using the CLI:
```bash
faststream run basic:app
```

Additional CLI options:

*   `--reload`: Enables hot reloading.
*   `--workers <num>`: Starts multiple worker processes for horizontal scaling.

Learn more about the CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream generates [AsyncAPI](https://www.asyncapi.com/) documentation for your project automatically.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependency Injection

FastStream uses a dependency management system inspired by `pytest fixtures` and `FastAPI Depends`.

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

## Framework Integrations

### Any Framework

Use FastStream brokers with any framework by starting and stopping them according to your application's lifespan.

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

Integrate FastStream into FastAPI with the `KafkaRouter` (or other broker router):

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

*   ⭐ Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   💬 Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   🗣️ Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>