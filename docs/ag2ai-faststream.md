# FastStream: Build Powerful and Efficient Microservices with Ease

**Effortlessly integrate event streams into your services with FastStream, the Python framework designed for modern microservices.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
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

## Key Features of FastStream

FastStream simplifies event stream integration, allowing you to build robust and scalable microservices with ease.  Visit the [original repository](https://github.com/ag2ai/faststream) to learn more.

*   **Multiple Broker Support:**  Seamlessly work with popular message brokers, including [Kafka](https://kafka.apache.org/), [RabbitMQ](https://www.rabbitmq.com/), [NATS](https://nats.io/), and [Redis](https://redis.io/).
*   **Pydantic Validation:**  Utilize [Pydantic's](https://docs.pydantic.dev/) powerful validation to ensure data integrity and type safety for incoming and outgoing messages.
*   **Automatic Documentation:**  Generate and maintain comprehensive [AsyncAPI](https://www.asyncapi.com/) documentation automatically, simplifying service integration and understanding.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in development and improving code quality.
*   **Dependency Injection:**  Manage service dependencies efficiently using FastStream's built-in dependency injection system.
*   **Testability:**  Leverage in-memory testing capabilities, making your CI/CD pipeline faster and more reliable.
*   **Extensibility:**  Enhance functionality with extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:**  Integrate FastStream with various HTTP frameworks, especially [FastAPI](https://fastapi.tiangolo.com/), for flexible application development.

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## History

FastStream builds upon the experience and ideas from [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), providing a unified approach for handling streamed data regardless of the underlying protocol.

---

## Installation

FastStream is compatible with **Linux**, **macOS**, **Windows**, and most **Unix**-style operating systems. Install using `pip`:

```bash
pip install 'faststream[kafka]'  # For Kafka support
# or
pip install 'faststream[rabbit]'  # For RabbitMQ support
# or
pip install 'faststream[nats]'    # For NATS support
# or
pip install 'faststream[redis]'   # For Redis support
```

By default **FastStream** uses **PydanticV2** written in **Rust**, but you can downgrade it manually, if your platform has no **Rust** support - **FastStream** will work correctly with **PydanticV1** as well.

---

## Writing Application Code

FastStream utilizes function decorators (`@broker.subscriber` and `@broker.publisher`) to simplify message queue interactions, including:

*   Consuming and producing data to event queues
*   Decoding and encoding JSON-encoded messages

These decorators allow you to focus on core application logic, simplifying development.  FastStream also integrates with [Pydantic](https://docs.pydantic.dev/) for easy data serialization and validation.

Here is an example Python app using **FastStream** that consumes data from an incoming data stream and outputs the data to another one:

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

You can use [Pydantic’s [`BaseModel`](https://docs.pydantic.dev/usage/models/)](https://docs.pydantic.dev/usage/models/) to define messages with a declarative syntax:

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

Use `TestBroker` context managers for in-memory testing.  This allows you to test your application quickly without a running broker.

Example tests using `pytest`:

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

Use the FastStream CLI to run your application.

1.  **Install the CLI:**

    ```bash
    pip install "faststream[cli]"
    ```

2.  **Run your application:**

    ```bash
    faststream run basic:app
    ```

    You should see output indicating the app has started.

3.  **Hot Reloading:**

    ```bash
    faststream run basic:app --reload
    ```

4.  **Multiprocessing:**

    ```bash
    faststream run basic:app --workers 3
    ```

Learn more about the CLI [here](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream automatically generates documentation using the [AsyncAPI](https://www.asyncapi.com/) specification, streamlining service integration.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream employs a dependency management system (leveraging [FastDepends](https://lancetnik.github.io/FastDepends/)) similar to `pytest fixtures` and `FastAPI Depends`.

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

## HTTP Framework Integrations

### Any Framework

Use `MQBrokers` independently within your application, starting and stopping them according to your application's lifecycle.

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

Integrate FastStream with FastAPI using `StreamRouter` and the `@router.subscriber` and `@router.publisher` decorators.

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

Find more integration details [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/).

---

## Stay Connected

*   [GitHub Repository](https://github.com/ag2ai/faststream/) - Star us!
*   [EN Discord Server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram Group](https://t.me/python_faststream)

---

## Contributors

Thanks to all the contributors who have helped make this project better!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>