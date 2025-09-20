# FastStream: Effortless Event Stream Integration for Microservices

**Streamline your microservices with FastStream, the Python framework that makes integrating with message brokers a breeze.** ([Original Repo](https://github.com/ag2ai/faststream))

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/Month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream:

*   **Multiple Broker Support:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis, simplifying your event streaming architecture.
*   **Pydantic Validation:** Leverage Pydantic for robust data validation and serialization of incoming messages, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:**  Generate and share clear, up-to-date documentation of your services automatically, saving time and improving collaboration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early and improving your development experience.
*   **Dependency Injection:** Manage service dependencies efficiently using FastStream's built-in dependency injection system.
*   **Simplified Testing:** Test your services with ease using in-memory tests, accelerating your CI/CD pipeline and improving reliability.
*   **Extensible Architecture:** Customize lifespans, serialization, and add middleware to match the specific needs of your applications.
*   **Framework Agnostic:** Works seamlessly with any HTTP framework, including a dedicated FastAPI plugin for streamlined integration.

---

## Getting Started

### Installation

Install FastStream using pip with the broker you need:

```bash
pip install 'faststream[kafka]'  # For Kafka
pip install 'faststream[rabbit]' # For RabbitMQ
pip install 'faststream[nats]'   # For NATS
pip install 'faststream[redis]'  # For Redis
```

### Writing app code

FastStream simplifies consuming and producing data to Event queues.

Here is an example Python app:

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

### Testing the service

The service can be tested using the `TestBroker` context managers:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1,}, "in")

@pytest.mark.asyncio
async def test_invalid():
    async with TestKafkaBroker(broker) as br:
        with pytest.raises(pydantic.ValidationError):
            await br.publish("wrong message", "in")
```

### Running the application

Install FastStream CLI:

```bash
pip install "faststream[cli]"
```

Run the service using the command:

```shell
faststream run basic:app
```

You can learn more about **CLI** features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Documentation

Explore the full documentation for comprehensive guides, tutorials, and API references.

*   [FastStream Documentation](https://faststream.ag2.ai/latest/)

---

## History

FastStream builds on the experiences from FastKafka and Propan, offering a unified approach to processing streamed data. New development focuses on this project.

---

## Project Documentation

FastStream automatically generates project documentation according to the [**AsyncAPI**](https://www.asyncapi.com/) specification.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream (thanks to [**FastDepends**](https://lancetnik.github.io/FastDepends/)) has a dependency management system similar to `pytest fixtures` and `FastAPI Depends` at the same time.

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

---

## Stay Connected

*   Give us a star on [GitHub](https://github.com/ag2ai/faststream/)
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thank you to all the contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>