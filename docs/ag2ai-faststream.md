# FastStream: Effortlessly Integrate Event Streams into Your Microservices

**Build robust, data-driven microservices faster with FastStream, the Python framework designed for seamless event stream integration.** ([Back to Repo](https://github.com/ag2ai/faststream))

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
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   **Multiple Broker Support:** Integrate seamlessly with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Validate incoming messages using Pydantic's powerful data validation capabilities, ensuring data integrity.
*   **Automatic AsyncAPI Docs:** Generate documentation automatically, simplifying service integration and providing clear communication.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early and enhancing the developer experience.
*   **Dependency Injection:** Manage dependencies efficiently with FastStream's built-in dependency injection system.
*   **Simplified Testing:** Test your streaming services easily with in-memory tests, speeding up your CI/CD pipeline and improving reliability.
*   **Extensible Architecture:** Customize your streaming applications with extensions for lifespans, custom serialization, and middleware.
*   **Flexible Integrations:** Integrate seamlessly with any HTTP framework, especially FastAPI, for unified application development.

---

## Get Started

*   **Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

## How FastStream Works

FastStream simplifies the development of event-driven microservices by abstracting the complexities of message queue interaction. It offers an easy-to-use API for producers and consumers, handling parsing, networking, and documentation generation automatically.

### Key Benefits:

*   **Reduced Complexity:** Focus on your business logic, not the underlying messaging infrastructure.
*   **Faster Development:** Accelerate your development cycle with features like automatic documentation and in-memory testing.
*   **Improved Reliability:** Ensure data integrity with Pydantic validation and robust error handling.
*   **Increased Productivity:** Leverage built-in features like dependency injection and intuitive type hinting.

## Core Concepts

FastStream uses function decorators to make the process of consuming and producing data to event queues easier. You can specify the processing logic for consumers and producers, allowing you to focus on the core business logic of your application without worrying about the underlying integration.

### Example

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

## Testing Made Easy

FastStream streamlines testing with `TestBroker` context managers. It redirects your subscriber and publisher functions to in-memory brokers. This approach avoids the need for a running broker and dependencies.

Here's a test example:

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

## Running Your Application

To run your FastStream application:

1.  **Install the CLI:** `pip install "faststream[cli]"`
2.  **Run the service:** `faststream run basic:app`

### Additional CLI Features:

*   **Hot Reload:** `faststream run basic:app --reload`
*   **Multiprocessing:** `faststream run basic:app --workers 3`

Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## History

**FastStream** builds upon the strengths of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan). By combining the best aspects of both, FastStream provides a unified approach for building services that process streamed data, regardless of the underlying protocol.

---

## Installation

**FastStream** is compatible with Linux, macOS, Windows, and most Unix-style operating systems. Install it using `pip`:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

By default, **FastStream** uses **PydanticV2** written in **Rust**, but you can downgrade it manually, if your platform has no **Rust** support - **FastStream** will work correctly with **PydanticV1** as well.

---

## Dependencies

**FastStream** (thanks to [**FastDepends**](https://lancetnik.github.io/FastDepends/)) has a dependency management system similar to `pytest fixtures` and `FastAPI Depends` at the same time. Function arguments declare which dependencies you want are needed, and a special decorator delivers them from the global Context object.

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

**FastStream** is fully compatible with FastAPI.

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

## Stay Connected

*   Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star!
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>