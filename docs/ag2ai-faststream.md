# FastStream: Effortlessly Integrate Event Streams into Your Microservices

**Simplify your event-driven architecture with FastStream, the Python framework designed for seamless integration with message brokers.**

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

*   **Unified API for Multiple Brokers:** Connect effortlessly to Kafka, RabbitMQ, NATS, and Redis with a single, consistent interface.
*   **Pydantic Validation:** Seamlessly integrate data validation with Pydantic for robust message handling and data integrity.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation automatically to streamline service integration and understanding.
*   **Intuitive Development Experience:** Benefit from full-typed editor support that catches errors early, leading to smoother development cycles.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in dependency injection system.
*   **Simplified Testing:** Utilize in-memory tests for faster, more reliable CI/CD pipelines.
*   **Extensibility:** Customize your streaming workflows with extensions for lifespans, custom serialization, and middleware.
*   **Framework Integration:** Integrate with any HTTP framework, particularly FastAPI, for flexible and versatile application development.

---

**Dive deeper into the documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## Why Choose FastStream?

FastStream simplifies the complexities of building event-driven microservices. It offers a streamlined developer experience, providing a robust foundation for both beginners and experienced developers alike. Whether you're starting a new project or looking to scale, FastStream equips you with the tools you need to excel in the world of data streaming.

---

## History

FastStream is the evolution of experiences from FastKafka and Propan, created to provide a unified way to process streamed data. The development focus is now on this project, offering the best features for your streaming needs.

---

## Installation

**FastStream** supports **Linux**, **macOS**, **Windows**, and most **Unix**-style operating systems. Install it using `pip`:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

By default, **FastStream** uses **PydanticV2**, but you can downgrade it manually, if your platform has no **Rust** support (**PydanticV1**).

---

## Quickstart: Writing App Code

FastStream simplifies consuming and producing data through function decorators: `@broker.subscriber` and `@broker.publisher`. They handle parsing, networking, and documentation. Use Pydantic for input data validation.

Here's an example:

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

You can define messages with Pydantic's `BaseModel`:

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

Test your service efficiently with the `TestBroker` context manager. It redirects messages to in-memory brokers, enabling rapid testing without a running broker.

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

## Running the Application

Start your FastStream application using the built-in CLI.

First, install the CLI:

```bash
pip install "faststream[cli]"
```

Then, run your service:

```bash
faststream run basic:app
```

Enhance your development with hot reload:

```bash
faststream run basic:app --reload
```

or horizontal scaling with multiprocessing:

```bash
faststream run basic:app --workers 3
```

Find more on CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation based on the AsyncAPI specification. This documentation simplifies service integration by clearly defining channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency management system like `pytest fixtures` and `FastAPI Depends`, utilizing function arguments for dependency declarations.

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

## Integrations with HTTP Frameworks

### Any Framework

Use `MQBrokers` independently of a `FastStream` app by starting and stopping them according to your application's lifespan.

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

Integrate FastStream with FastAPI via `StreamRouter`:

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

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

Your support helps us improve and develop FastStream. Thank you!

---

## Contributors

A special thanks to our contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>