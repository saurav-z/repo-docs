# FastStream: Effortlessly Build & Integrate Streaming Microservices

**Streamline your microservice development with FastStream, the Python framework for effortless event stream integration.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Validate incoming messages with powerful Pydantic data validation.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive API documentation automatically.
*   **Intuitive Development Experience:** Benefit from full-typed editor support, catching errors early.
*   **Dependency Injection:** Manage service dependencies efficiently with a built-in DI system.
*   **Testability:** Simplify testing with in-memory test brokers for fast and reliable CI/CD.
*   **Extensibility:** Extend functionality with lifespans, custom serialization, and middleware.
*   **Framework Integrations:** Integrate with any HTTP framework, especially optimized for FastAPI.

Learn more in the [FastStream Documentation](https://faststream.ag2.ai/latest/).

---

## What is FastStream?

FastStream simplifies building streaming microservices by handling the complexities of message queue integration. It provides a unified API and automates parsing, networking, and documentation generation, making it easier for developers of all skill levels to create robust, data-driven applications. Whether you're new to streaming or looking to scale, FastStream has you covered.

---

## Getting Started

### Installation

Install FastStream using pip:

```bash
pip install 'faststream[kafka]'  # For Kafka
pip install 'faststream[rabbit]' # For RabbitMQ
pip install 'faststream[nats]'   # For NATS
pip install 'faststream[redis]'  # For Redis
```

### Example Usage

Here's a simple example to get you started:

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

For detailed usage, refer to the documentation or original [FastStream Repository](https://github.com/ag2ai/faststream).

---

## Why FastStream?

FastStream is a new project built from the learnings of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan). It offers a unified and efficient way to process streamed data, providing a superior experience for building modern microservices.

---

## Testing

FastStream simplifies testing with its `TestBroker` context managers. Test your services easily using pytest:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1}, "in")
```

---

## Running Your Application

Use the FastStream CLI to run your application:

```bash
pip install "faststream[cli]"
faststream run basic:app
```

For hot reloading and multiprocessing scaling:

```bash
faststream run basic:app --reload --workers 3
```

Learn more about **CLI** features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation based on the [AsyncAPI](https://www.asyncapi.com/) specification. This significantly simplifies service integration by providing clear documentation of channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency management system similar to `pytest fixtures` and `FastAPI Depends` based on [**FastDepends**](https://lancetnik.github.io/FastDepends/)

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

## Integration with HTTP Frameworks

### Any Framework

You can use **FastStream** `MQBrokers` without a `FastStream` application.
Just *start* and *stop* them according to your application's lifespan.

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

Also, **FastStream** can be used as part of **FastAPI**.

Just import a **StreamRouter** you need and declare the message handler with the same `@router.subscriber(...)` and `@router.publisher(...)` decorators.

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

More integration features can be found [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](https://github.com/ag2ai/faststream/blob/main/CONTRIBUTING.md) for details.

---

## Stay Connected

*   Give the project a star on [GitHub](https://github.com/ag2ai/faststream/).
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

---

## Acknowledgements

A big thank you to all of our [contributors](https://github.com/ag2ai/faststream/graphs/contributors)!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>