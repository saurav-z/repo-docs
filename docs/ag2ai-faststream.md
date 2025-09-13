# FastStream: Effortless Event Stream Integration for Microservices

**Streamline your microservices with FastStream, a Python framework that simplifies event-driven architecture and boosts developer productivity.** ([Original Repository](https://github.com/ag2ai/faststream))

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
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

*   **Multiple Broker Support:**  Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:**  Utilize Pydantic for robust data validation and serialization of incoming messages.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation for your services automatically.
*   **Intuitive Development Experience:** Benefit from full-typed editor support, catching errors early in the development process.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Simplified Testing:**  Test your code with in-memory tests.
*   **Extensibility:** Use extensions for lifespans, custom serialization and middleware
*   **Flexible Integrations:** Compatible with any HTTP framework, with a dedicated FastAPI plugin.

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## What is FastStream?

FastStream is a Python framework designed to simplify the development of event-driven microservices.  It streamlines the process of building producers and consumers for message queues, handling parsing, networking, and documentation generation automatically. Based on the successes of FastKafka and Propan, FastStream provides a unified and efficient way to build scalable services that process streamed data.

---

## Installation

FastStream supports Linux, macOS, Windows, and most Unix-like operating systems. Install it using pip:

```bash
pip install 'faststream[kafka]'  # For Kafka support
# or
pip install 'faststream[rabbit]' # For RabbitMQ support
# or
pip install 'faststream[nats]'   # For NATS support
# or
pip install 'faststream[redis]'  # For Redis support
```

*By default, FastStream uses PydanticV2 (written in Rust). However, if your platform doesn't support Rust, FastStream seamlessly falls back to PydanticV1.*

---

## Getting Started: Writing App Code

FastStream utilizes function decorators (`@broker.subscriber` and `@broker.publisher`) to simplify the creation of producers and consumers.  These decorators handle the nuances of consuming/producing data to event queues, including JSON encoding/decoding.

Leverage Pydantic for straightforward input data parsing using type annotations.

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

You can also define messages with Pydantic's `BaseModel` for structured data:

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

FastStream provides `TestBroker` context managers to enable in-memory testing. This allows for rapid, dependency-free testing of your services using popular testing frameworks like `pytest`.

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

To run your FastStream application, use the FastStream CLI. First, install the CLI:

```bash
pip install "faststream[cli]"
```

Then, run your application with the command:

```bash
faststream run basic:app
```

For enhanced development workflows, utilize the `--reload` and `--workers` options:

```bash
faststream run basic:app --reload  # Enable hot reload
faststream run basic:app --workers 3 # Enable multiprocessing
```

Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation for your project using the AsyncAPI specification.  This documentation simplifies the integration of services by clearly outlining message formats and channels.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream features a robust dependency management system (leveraging FastDepends) similar to `pytest fixtures` and `FastAPI Depends`.  Define dependencies within function arguments and utilize the special decorator to manage them.

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

### General Framework Integration

Integrate `MQBrokers` into any application with the `start` and `stop` methods.

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

FastStream integrates seamlessly with FastAPI.

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
Find additional integration features [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

Support FastStream and stay informed:

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [Telegram group](https://t.me/python_faststream)

---

## Contributors

A big thank you to our amazing contributors:

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>