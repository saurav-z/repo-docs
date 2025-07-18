# FastStream: Effortlessly Build Event-Driven Microservices

**FastStream streamlines event stream integration for your services, empowering you to build robust and scalable microservices with ease.**

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

*   **Multiple Broker Support:** Seamlessly integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Utilize Pydantic for data validation and serialization, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation automatically to easily understand your service's message formats and channels.
*   **Intuitive Development Experience:** Enjoy full-typed editor support for faster development and error detection.
*   **Powerful Dependency Injection:** Simplify service management with FastStream's built-in dependency injection system.
*   **Simplified Testing:** Leverage in-memory tests for rapid and reliable CI/CD pipelines.
*   **Extensibility:** Use extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:** FastStream is fully compatible with any HTTP framework and is especially easy to integrate with FastAPI.

---

## Overview

FastStream simplifies the complexities of building event-driven microservices. It handles the intricacies of message queue interactions, parsing, and documentation generation. Designed with both novice and experienced developers in mind, FastStream accelerates your development process while offering flexibility for advanced use cases.  This project builds upon the knowledge and experience of [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), inheriting the best features from both.

For more information, check out the [FastStream Documentation](https://faststream.ag2.ai/latest/).

---

## Installation

FastStream supports Linux, macOS, Windows, and most Unix-style operating systems. Install it easily with pip:

```bash
pip install 'faststream[kafka]' # or rabbit, nats, redis
```

By default, FastStream utilizes PydanticV2 for optimal performance. However, if your platform lacks Rust support, you can use PydanticV1 without issues.

---

## Getting Started: Writing App Code

FastStream simplifies consuming and producing data via convenient decorators like `@broker.subscriber` and `@broker.publisher`, making it easy to focus on the core logic of your application.  It also integrates with Pydantic for seamless data serialization using type annotations.

Here's a simple example:

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

You can also use Pydantic's `BaseModel` for defining your message structures:

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

Test your FastStream service efficiently using `TestBroker`, which redirects broker calls to in-memory brokers.

Example test using pytest:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1}, "in")

@pytest.mark.asyncio
async def test_invalid():
    async with TestKafkaBroker(broker) as br:
        with pytest.raises(pydantic.ValidationError):
            await br.publish("wrong message", "in")
```

---

## Running Your Application

Use the FastStream CLI to run your application:

1.  **Install CLI:** `pip install "faststream[cli]"`
2.  **Run your app:** `faststream run basic:app`

For hot-reloading and multiprocessing, try:

```bash
faststream run basic:app --reload
faststream run basic:app --workers 3
```

Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation based on the [AsyncAPI](https://www.asyncapi.com/) specification.  This simplifies service integration by providing clear documentation of channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream offers a dependency management system similar to `pytest fixtures` and `FastAPI Depends`, simplifying how you handle dependencies.

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

Use FastStream's `MQBrokers` independently of a `FastStream` application by starting and stopping them according to your application's lifecycle.

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

Easily integrate FastStream with FastAPI using a `StreamRouter`:

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

Show your support and stay in touch!

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

A huge thank you to all of our contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>
```
Key improvements and explanations:

*   **SEO Optimization:** The updated README includes keywords like "event-driven," "microservices," and names of supported message brokers to help with search engine optimization.
*   **Clear Headings:**  Organized content with clear and concise headings and subheadings.
*   **Concise Summary/Hook:**  Replaced the original heading with a more engaging and descriptive introduction that clearly states the purpose of FastStream.
*   **Bulleted Key Features:**  Uses a bulleted list to make the key features easily scannable and highlights the benefits.
*   **Detailed Explanations:** Provides more detail on how to install, write the code, test, and run the application.
*   **Added Links:** Added links to all referenced external projects and websites.
*   **Structure:**  The format is well-organized for easy reading and understanding.
*   **Conciseness:**  Removed redundant phrases and condensed information while keeping the core message.
*   **Call to Action:** The "Stay Connected" section encourages community engagement.