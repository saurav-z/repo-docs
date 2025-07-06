# FastStream: Effortlessly Integrate Event Streams for Your Microservices

FastStream simplifies event stream integration, making building modern, data-centric microservices easier than ever. Check out the [original repo](https://github.com/ag2ai/faststream/) for more details.

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
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.airt.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support:** Seamlessly work with [**Kafka**](https://kafka.apache.org/), [**RabbitMQ**](https://www.rabbitmq.com/), [**NATS**](https://nats.io/), and [**Redis**](https://redis.io/).
*   **Pydantic Validation:** Validate messages using [**Pydantic**](https://docs.pydantic.dev/) for robust data handling.
*   **Automatic Documentation:** Generate [**AsyncAPI**](https://www.asyncapi.com/) documentation automatically to simplify service integration.
*   **Type-Safe Development:** Benefit from full-typed editor support, catching errors early in the development cycle.
*   **Dependency Injection:** Manage dependencies efficiently with FastStream's built-in dependency injection system.
*   **Simplified Testing:** Utilize in-memory tests to speed up your CI/CD pipeline and ensure reliability.
*   **Extensibility:** Leverage extensions for lifespans, custom serialization, and middleware to tailor FastStream to your needs.
*   **Framework Compatibility:** Integrate with any HTTP framework, with dedicated support for [**FastAPI**](#fastapi-plugin).

---

## How it Works

FastStream simplifies the creation of message producers and consumers, handling parsing, networking, and documentation generation automatically. This streamlined approach is designed with developers in mind, making it easy to build efficient and scalable streaming microservices.

## Installation

Install FastStream with your preferred broker dependencies:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream uses PydanticV2, written in Rust, by default. If Rust isn't supported on your platform, FastStream will work with PydanticV1.

## Core Concepts

### Writing App Code

Use function decorators (`@broker.subscriber` and `@broker.publisher`) to define consumer and producer logic. Focus on your core business logic while FastStream handles the underlying complexities of event queueing.

### Pydantic Integration

Utilize [**Pydantic**](https://docs.pydantic.dev/) for seamless data validation and serialization. Type annotations streamline input message handling, ensuring structured data in your applications.

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

You can also define messages using Pydantic's `BaseModel` class for a declarative syntax.

### Testing Your Service

Test your service effectively using `TestBroker`. TestBroker redirects functions to InMemory brokers, letting you test without a running broker.

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

### Running the Application

Run your FastStream application with the CLI after installing the `faststream[cli]` package:

```shell
faststream run basic:app
```

Enhance your development experience with hot reload (`--reload`) and multiprocessing (`--workers <num>`).

## Project Documentation

FastStream automatically generates documentation adhering to the [**AsyncAPI**](https://www.asyncapi.com/) specification.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

## Dependencies

FastStream uses `FastDepends` to manage dependencies. Function arguments declare the dependencies needed, and the decorator delivers them from the global Context object.

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

## HTTP Frameworks Integrations

### Any Framework

Use FastStream `MQBrokers` without a `FastStream` application. Start and stop them according to your application's lifespan.

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

Integrate FastStream with FastAPI by importing a `StreamRouter`. Use the `@router.subscriber(...)` and `@router.publisher(...)` decorators for message handling.

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

## Stay Connected

*   Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>