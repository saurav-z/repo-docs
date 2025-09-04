# FastStream: Effortlessly Integrate Event Streams in Your Microservices

**Simplify your event-driven architecture and build robust, scalable microservices with FastStream, the Python framework that makes message queue integration a breeze.  [Check out the FastStream repository](https://github.com/ag2ai/faststream/) for more details!**

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

*   **Multi-Broker Support:** Integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis through a unified API.
*   **Pydantic Validation:**  Ensure data integrity with seamless integration of Pydantic for message validation and serialization.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation automatically to streamline service integration and understanding.
*   **Type Hinting & Developer Experience:** Benefit from robust type hinting support for enhanced code quality, maintainability, and a smooth developer experience.
*   **Dependency Injection:**  Manage your service dependencies with a powerful built-in dependency injection system.
*   **Simplified Testing:**  Easily test your services with in-memory tests, accelerating your CI/CD pipeline and improving reliability.
*   **Extensibility:** Leverage extensions for lifespans, custom serialization, and middleware to tailor FastStream to your specific needs.
*   **Framework Agnostic:** Seamlessly integrates with any HTTP framework, including FastAPI.

---

**[Explore the FastStream Documentation](https://faststream.ag2.ai/latest/)**

---

## What is FastStream?

FastStream is a Python framework designed to simplify the development of microservices that interact with event streams. Built upon the proven concepts of FastKafka and Propan, FastStream provides a streamlined approach to building data-driven applications, focusing on ease of use without sacrificing advanced capabilities.  Whether you're new to streaming or an experienced developer, FastStream offers a robust and efficient solution.

---

## Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-based systems. Install it using pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream utilizes PydanticV2 written in Rust, but will gracefully fall back to PydanticV1 if Rust isn't supported on your platform.

---

## Getting Started: Writing App Code

FastStream simplifies message queue integration with convenient decorators: `@broker.subscriber` for consuming messages and `@broker.publisher` for publishing messages.  These decorators handle message parsing, networking, and documentation generation, allowing you to focus on your core business logic.

FastStream leverages Pydantic for type-safe data parsing.

Example:

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

You can define messages using Pydantic's `BaseModel` for structured data:

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

FastStream provides a `TestBroker` context manager for easy in-memory testing.

Example using pytest:

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

Use the FastStream CLI to run your application:

1.  Install the CLI:

    ```bash
    pip install "faststream[cli]"
    ```

2.  Run your app:

    ```bash
    faststream run basic:app
    ```

    Enhance your development workflow with hot reloading and multiprocessing:

    ```bash
    faststream run basic:app --reload
    faststream run basic:app --workers 3
    ```

    [Learn more about CLI features](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation in the [AsyncAPI](https://www.asyncapi.com/) specification. This documentation simplifies service integration by clearly outlining channels and message formats, and it's created automatically!

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependency Injection

FastStream's dependency injection system, similar to `pytest fixtures` and `FastAPI Depends`, allows you to easily manage dependencies:

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

FastStream `MQBrokers` can be used independently.

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

[Find more integration features here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

Support FastStream and stay informed:

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thanks to all contributors:

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>