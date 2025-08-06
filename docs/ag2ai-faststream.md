# FastStream: Build Modern, Data-Driven Microservices with Ease

Effortlessly integrate event streams into your services with **FastStream**, a powerful Python framework.

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)

[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads per Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)

[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)

[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)

[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   **Multiple Broker Support:** Seamlessly work with various message brokers, including Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization of incoming messages.
*   **Automatic AsyncAPI Docs:** Generate up-to-date documentation to simplify service integration using the AsyncAPI specification.
*   **Intuitive Development:** Experience a smooth development workflow with full-typed editor support, catching errors early.
*   **Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Simplified Testing:** Built-in testing with in-memory brokers make testing your services fast and reliable.
*   **Extensible Architecture:** Easily extend functionality with extensions for lifespans, custom serialization, and middleware.
*   **Flexible Integrations:** Compatible with any HTTP framework, and fully integrated with FastAPI.

**Get started with FastStream today and simplify your microservice architecture!**

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## How FastStream Came to Be

FastStream is a modern framework that builds on the experiences gained from FastKafka and Propan.  This project focuses on offering a unified approach for building services that process streamed data regardless of the underlying protocol. For new services, FastStream is the recommended choice.

---

## Installation

FastStream supports **Linux**, **macOS**, **Windows**, and most **Unix**-style operating systems. Install it using `pip`:

```bash
pip install 'faststream[kafka]'  # or
pip install 'faststream[rabbit]' # or
pip install 'faststream[nats]'   # or
pip install 'faststream[redis]'
```

FastStream defaults to **PydanticV2** for performance, but it will work with **PydanticV1** if you need to downgrade for compatibility.

---

## Writing App Code with FastStream

FastStream simplifies event stream integration with function decorators `@broker.subscriber` and `@broker.publisher`:

*   Consuming and producing data to event queues.
*   Decoding and encoding JSON-encoded messages.

Use type annotations and Pydantic for structured data:

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

Define message structures with Pydantic's `BaseModel`:

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

Test your service easily with `TestBroker` context managers.  This simplifies your testing workflow by allowing you to test your app logic without the need for a running broker.

Here's how you might test your service with pytest:

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

Run your application with the FastStream CLI.  First, install it:

```shell
pip install "faststream[cli]"
```

Then, run your app:

```shell
faststream run basic:app
```

Use hot reload:

```shell
faststream run basic:app --reload
```

Or, use multiprocessing:

```shell
faststream run basic:app --workers 3
```

Learn more about the CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation for your project using the [AsyncAPI](https://www.asyncapi.com/) specification.  This can simplify service integration.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency injection system similar to `pytest fixtures` and `FastAPI Depends` based on [FastDepends](https://lancetnik.github.io/FastDepends/).

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

Integrate FastStream with FastAPI using a `StreamRouter`.

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

## Stay Connected

*   Give us a star on [GitHub](https://github.com/ag2ai/faststream/).
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

Your support helps us grow and improve FastStream.

---

## Contributors

A big thanks to all the amazing contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>