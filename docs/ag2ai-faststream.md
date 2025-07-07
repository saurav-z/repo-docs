# FastStream: Effortlessly Build Event-Driven Microservices

FastStream is a Python framework designed to simplify the creation of event-driven microservices, enabling developers to build robust and scalable applications with ease. Check out the [original repo](https://github.com/ag2ai/faststream) for more information!

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
*   **Pydantic Validation:** Leverage Pydantic for data validation and serialization, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation for your event streams automatically.
*   **Intuitive Development:** Benefit from full-typed editor support for a smooth and error-free coding experience.
*   **Dependency Injection:** Manage dependencies efficiently with FastStream's built-in dependency injection system.
*   **Testable Services:** Utilize in-memory testing for faster and more reliable CI/CD pipelines.
*   **Extensible:** Customize your application with extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:** Easily integrate with various HTTP frameworks, including FastAPI.

---

**Documentation:** [https://faststream.airt.ai/latest/](https://faststream.airt.ai/latest/)

---

## History

FastStream is a modern framework built upon the experiences gained from FastKafka and Propan, merging the best features to provide a unified approach for building event-driven services, focusing on ease of use and efficiency.

---

## Installation

FastStream supports **Linux**, **macOS**, **Windows**, and most **Unix**-style operating systems. Install with pip:

```bash
pip install 'faststream[kafka]' # for Kafka
# or
pip install 'faststream[rabbit]' # for RabbitMQ
# or
pip install 'faststream[nats]' # for NATS
# or
pip install 'faststream[redis]' # for Redis
```

---

## Getting Started: Writing App Code

FastStream uses function decorators like `@broker.subscriber` and `@broker.publisher` to simplify message queue interactions, handling parsing, networking, and documentation generation.

### Example: Consuming and Producing Messages

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

### Using Pydantic Models

Define message structures using Pydantic's `BaseModel`:

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

Use `TestBroker` context managers for in-memory testing:

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

1.  **Install FastStream CLI:** `pip install "faststream[cli]"`
2.  **Run Your App:** `faststream run basic:app`
3.  **With Hot Reload:** `faststream run basic:app --reload`
4.  **With Multiprocessing:** `faststream run basic:app --workers 3`

Learn more about CLI features [here](https://faststream.airt.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates [AsyncAPI](https://www.asyncapi.com/) documentation for your project, simplifying service integration.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream utilizes a dependency management system, similar to pytest fixtures and FastAPI Depends, using function arguments and a special decorator to deliver dependencies.

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

Use `MQBrokers` independently:

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

Integrate with FastAPI using `StreamRouter`:

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

More integration features can be found [here](https://faststream.airt.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

*   Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Thank you to all the contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>