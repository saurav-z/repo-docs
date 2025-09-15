# FastStream: Effortlessly Integrate Event Streams with Python 🚀

**Simplify your microservices with FastStream, the Python framework for building robust and scalable event-driven applications.** ([Original Repo](https://github.com/ag2ai/faststream))

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Tests Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)
[![Downloads per Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI Package Version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

FastStream empowers developers to build efficient and scalable event-driven architectures. Here's what makes it stand out:

*   **Multiple Broker Support:** Seamlessly integrate with popular message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Leverage Pydantic for robust data validation and serialization, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Automatically generate documentation for your services, simplifying integration and collaboration.
*   **Developer-Friendly:** Benefit from full-typed editor support, making your development experience smooth and error-free.
*   **Dependency Injection:** Manage dependencies effectively with FastStream's built-in dependency injection system.
*   **Simplified Testing:** Utilize in-memory testing for fast and reliable CI/CD pipelines.
*   **Extensibility:** Extend functionality with extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:** Compatible with any HTTP framework, including seamless integration with FastAPI.

---

## Documentation

Explore comprehensive documentation to get started quickly: [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## History

FastStream builds upon the strengths of FastKafka and Propan. This new project brings the best features from both projects, creating a unified way to write streaming services. If you're starting a new service, FastStream is the recommended approach.

---

## Installation

Install FastStream with pip, supporting Linux, macOS, Windows, and Unix-like operating systems.

```bash
pip install 'faststream[kafka]'  # Install with Kafka support
# or
pip install 'faststream[rabbit]' # Install with RabbitMQ support
# or
pip install 'faststream[nats]'   # Install with NATS support
# or
pip install 'faststream[redis]'  # Install with Redis support
```

FastStream defaults to PydanticV2 (Rust-based) for performance. If your platform lacks Rust support, you can still use PydanticV1.

---

## Getting Started

FastStream simplifies event streaming with function decorators: `@broker.subscriber` for consuming and `@broker.publisher` for producing messages.

### Writing App Code

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

### Pydantic for Message Definition

Define your messages with Pydantic models for type safety and validation:

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

## Running Your Application

Run your FastStream application using the CLI:

1.  Install the CLI:  `pip install "faststream[cli]"`
2.  Run your app: `faststream run basic:app`

Use `--reload` for hot-reloading and `--workers` for multiprocessing.  Learn more in the [CLI documentation](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream generates AsyncAPI documentation, simplifying service integration:

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses `FastDepends` for dependency management, similar to pytest fixtures and FastAPI Depends.

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

Support the project and stay updated:

*   ⭐ Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   💬 Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   🗣️ Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thanks to all contributors!

[![Contributors](https://contrib.rocks/image?repo=ag2ai/faststream)](https://github.com/ag2ai/faststream/graphs/contributors)