# FastStream: Effortlessly Build Real-time, Data-Driven Microservices

**Simplify your event streaming integrations and build powerful, scalable microservices with FastStream.** ([Back to top](#faststream-effortlessly-build-real-time-data-driven-microservices))

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

*   **Multi-Broker Support:** Seamlessly integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:**  Validate message data using Pydantic for robust and reliable data handling.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive API documentation automatically to simplify service integration.
*   **Developer-Friendly:** Benefit from full-typed editor support, minimizing errors during development.
*   **Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Testability:** Utilize in-memory testing to accelerate CI/CD pipelines and ensure code quality.
*   **Extensibility:** Leverage extensions for lifespans, custom serialization, and middleware to customize your applications.
*   **Framework Agnostic:** Integrate easily with any HTTP framework, with native support for FastAPI.

[**Get started with FastStream today!**](https://github.com/ag2ai/faststream)

---

## Why FastStream?

FastStream simplifies the complexities of building microservices that leverage event streaming.  It allows you to focus on business logic by automating parsing, networking, and documentation generation. It's designed to empower developers of all experience levels, making it easier to create modern, data-centric applications.

---

## Installation

FastStream is compatible with various operating systems, including Linux, macOS, Windows, and most Unix-style systems. Install with pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

Note: While FastStream uses PydanticV2 for improved performance, you can manually downgrade to PydanticV1 if your platform doesn't support Rust. FastStream will still function correctly.

---

## Writing Application Code

FastStream provides convenient decorators, `@broker.subscriber` and `@broker.publisher`, to streamline message consumption and production. These decorators simplify the process of:

-   Consuming and producing data to event queues
-   Decoding and encoding JSON-encoded messages

This approach simplifies the development process, allowing you to focus on the core functionality of your application without dealing with low-level integration details.

Also, FastStream uses Pydantic to handle input JSON-encoded data, converting it to Python objects. Use type annotations to easily serialize your input messages.

Here's an example of a FastStream application that processes data from an incoming stream and outputs it to another:

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker
# from faststream.rabbit import RabbitBroker
# from faststream.nats import NatsBroker
# from faststream.redis import RedisBroker

broker = KafkaBroker("localhost:9092")
# broker = RabbitBroker("amqp://guest:guest@localhost:5672/")
# broker = NatsBroker("nats://localhost:4222/")
# broker = RedisBroker("redis://localhost:6379/")

app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

Additionally, Pydantic's `BaseModel` class facilitates defining messages with a declarative syntax, which helps specify message fields and their data types.

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

Use the `TestBroker` context managers for testing, which puts the Broker into "testing mode" by default.

The Tester redirects your `subscriber` and `publisher` decorated functions to the InMemory brokers, enabling rapid testing without a running broker and its dependencies.

Using pytest, a test for the service can be structured like this:

```python
# Code above omitted 👆

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

Start the application using the FastStream CLI command.

Before running the service, install the FastStream CLI:

```shell
pip install "faststream[cli]"
```

Run the service by providing the module (where your app implementation is located) and the app symbol:

```shell
faststream run basic:app
```

Example output:

```shell
INFO     - FastStream app starting...
INFO     - input_data |            - `HandleMsg` waiting for messages
INFO     - FastStream app started successfully! To exit press CTRL+C
```

FastStream offers a hot reload feature:

```shell
faststream run basic:app --reload
```

And also multiprocessing horizontal scaling:

```shell
faststream run basic:app --workers 3
```

Learn more about the CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream automatically generates documentation for your project according to the [AsyncAPI](https://www.asyncapi.com/) specification. You can utilize these generated artifacts to integrate your services and easily display web documentation for related teams.

These docs significantly simplify service integration as you can see the channels and message formats your application uses.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream incorporates a dependency management system similar to `pytest fixtures` and `FastAPI Depends` using [FastDepends](https://lancetnik.github.io/FastDepends/). Function arguments define dependencies, which are then delivered through a special decorator from the global Context object.

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

Use `MQBrokers` without a `FastStream` application. Just *start* and *stop* them according to your application's lifespan.

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

FastStream integrates with FastAPI.

Import the necessary `StreamRouter` and declare message handlers using the `@router.subscriber(...)` and `@router.publisher(...)` decorators.

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

Find more integration features [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/).

---

## Stay Connected

Show your support and stay updated:

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

Your support helps us maintain and improve FastStream. Thank you!

---

## Contributors

Thank you to the contributors who made this project better:

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>