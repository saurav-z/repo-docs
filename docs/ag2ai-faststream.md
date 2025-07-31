# FastStream: Effortlessly Integrate Event Streams in Your Microservices

**Simplify your event-driven architecture with FastStream, a Python framework designed to make building and managing message-based services a breeze.**

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
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

FastStream is a powerful framework designed for modern microservices, offering:

*   **Multiple Broker Support**: Seamlessly work with Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation**:  Validate incoming messages using Pydantic for robust data handling.
*   **Automatic AsyncAPI Docs**:  Generate comprehensive documentation automatically to simplify service integration.
*   **Intuitive Development**: Enjoy full-typed editor support for a smooth and error-free coding experience.
*   **Powerful Dependency Injection**: Efficiently manage service dependencies with FastStream's built-in DI system.
*   **Testable Services**:  Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility**:  Leverage extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations**:  Compatible with any HTTP framework, especially with the [FastAPI Plugin](#fastapi-plugin).

Ready to simplify your streaming microservices?  Get started with FastStream!

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## What is FastStream?

FastStream simplifies the development of event-driven microservices. It handles message parsing, networking, and documentation generation automatically, making it easier for developers of all skill levels to build scalable and resilient systems. This package is a recommended approach to create services capable of processing streamed data regardless of the underlying protocol.

---

## History

FastStream builds upon the experiences gained from FastKafka and Propan. It offers a unified way to process streamed data, regardless of the protocol. If you're starting a new service, this is the recommended package to use.

---

## Installation

FastStream supports Linux, macOS, Windows, and most Unix-style operating systems.

Install with pip:

```bash
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

By default FastStream uses PydanticV2 written in Rust, but you can downgrade it manually, if your platform has no Rust support - FastStream will work correctly with PydanticV1 as well.

---

## Writing App Code

FastStream provides convenient function decorators (`@broker.subscriber` and `@broker.publisher`) to handle message queue operations, decoding, and encoding JSON-encoded messages. This lets you focus on the core logic of your application.  

Leverage Pydantic for data validation and type safety.

**Example:**

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

Define messages with Pydantic's `BaseModel`:

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

Use `TestBroker` context managers for in-memory tests. This allows testing without a running broker.

**Example using pytest:**

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

Use the FastStream CLI for easy application startup:

1.  **Install the CLI:**

    ```bash
    pip install "faststream[cli]"
    ```

2.  **Run your service:**

    ```bash
    faststream run basic:app
    ```

    You'll see output indicating your app is running.
3.  **Hot Reload and Multiprocessing:**
    ```bash
    faststream run basic:app --reload
    faststream run basic:app --workers 3
    ```

    For more CLI information, see [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates [AsyncAPI](https://www.asyncapi.com/) documentation, enabling you to see which channels and message formats an application uses.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency management system similar to `pytest fixtures` and `FastAPI Depends` (thanks to FastDepends):

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

## Framework Integrations

### Any Framework

Use FastStream `MQBrokers` with any framework; start and stop them with your application's lifecycle:

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

Use FastStream with FastAPI by importing a `StreamRouter`:

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

*   [GitHub Repository](https://github.com/ag2ai/faststream/) - Star us!
*   [EN Discord server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Special thanks to the amazing contributors:

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>
```
Key improvements and summaries:

*   **SEO Optimization:**  Added relevant keywords like "event stream," "microservices," "message broker," and "Python framework" throughout the text.
*   **One-Sentence Hook:**  Provided a clear, concise opening sentence to grab the reader's attention.
*   **Clear Headings:**  Organized the README with clear headings and subheadings for readability.
*   **Bulleted Key Features:**  Highlighted key features with bullet points for easy skimming.
*   **Concise Descriptions:** Simplified explanations while retaining essential information.
*   **Actionable Installation and Usage:**  Provided clear installation instructions and example code snippets.
*   **Emphasis on Benefits:** Highlighted the advantages of using FastStream (e.g., ease of use, testing, documentation).
*   **Improved Formatting:** Used bolding, and other Markdown formatting for better visual appeal.
*   **Removed Redundancy:** Eliminated unnecessary phrases and repeated information.
*   **Expanded History section:** Provided a brief summary of the project's origins.
*   **Internal linking:** Linked key features to relevant sections.