# FastStream: Effortlessly Integrate Event Streams for Your Microservices

**Simplify event stream integration and supercharge your microservices with FastStream – the Python framework designed for speed, ease, and scalability.**

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

FastStream simplifies event stream integration for your services, handling parsing, networking, and documentation generation automatically.  It's designed to make building modern, data-centric microservices easier than ever.

*   ✅ **Multiple Broker Support:** Seamlessly work with Kafka, RabbitMQ, NATS, and Redis.
*   ✅ **Pydantic Validation:** Utilize Pydantic for efficient message serialization and validation.
*   ✅ **Automatic AsyncAPI Docs:** Generate comprehensive documentation to visualize and share your message schema.
*   ✅ **Intuitive & Type-Safe:** Benefit from full-typed editor support, reducing runtime errors.
*   ✅ **Powerful Dependency Injection:**  Manage service dependencies efficiently.
*   ✅ **Simplified Testing:** Employ in-memory tests for faster and more reliable CI/CD.
*   ✅ **Extensible:** Utilize extensions for lifespans, custom serialization, and middleware.
*   ✅ **Framework Agnostic:** Compatible with any HTTP framework, including seamless FastAPI integration.

For more details and to get started, explore the comprehensive [FastStream documentation](https://faststream.ag2.ai/latest/).

---

## History

FastStream evolved from FastKafka and Propan, drawing the best from both projects to create a unified framework for processing streamed data regardless of the underlying protocol. New development focuses on this project.

---

## Installation

FastStream supports Linux, macOS, Windows, and most Unix-like operating systems.

Install using pip:

```bash
pip install 'faststream[kafka]'   # Kafka
pip install 'faststream[rabbit]'  # RabbitMQ
pip install 'faststream[nats]'    # NATS
pip install 'faststream[redis]'   # Redis
```

By default FastStream uses PydanticV2 written in Rust, but you can downgrade it manually, if your platform has no Rust support - FastStream will work correctly with PydanticV1 as well.

---

## Writing App Code

Use the `@broker.subscriber` and `@broker.publisher` decorators to define message consumers and producers, streamlining data processing and message handling.

FastStream integrates Pydantic for effortless JSON serialization and validation using type annotations:

```python
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

FastStream includes `TestKafkaBroker` to test your applications without a running broker.

Example using pytest:

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

## Running Your Application

Start your FastStream application using the CLI after installation:

```bash
pip install "faststream[cli]"
faststream run basic:app
faststream run basic:app --reload  # Hot reload
faststream run basic:app --workers 3  # Multiprocessing scaling
```

Learn more about the CLI [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream generates [AsyncAPI](https://www.asyncapi.com/) documentation automatically for your project.  This simplifies service integration, letting you see what channels and message formats the application uses.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency injection system uses a similar approach to pytest fixtures and FastAPI Depends.

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

## Framework Integrations

### Any Framework

Use FastStream MQBrokers with any framework by starting and stopping them based on your application's lifecycle.

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

Integrate FastStream with FastAPI using StreamRouter:

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

Support FastStream and stay informed:

*   ⭐ [Star us on GitHub](https://github.com/ag2ai/faststream/)
*   💬 [Join our EN Discord server](https://discord.gg/qFm6aSqq59)
*   🗣️ [Join our RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thank you to everyone who has contributed!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>

[Back to Top](#faststream-effortlessly-integrate-event-streams-for-your-microservices)