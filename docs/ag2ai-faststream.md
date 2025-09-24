# FastStream: Effortlessly Build Modern, Data-Driven Microservices 🚀

**Tired of complex event stream integration?** FastStream simplifies building producers and consumers for message queues, automating parsing, networking, and documentation generation, so you can focus on your core business logic.  [Check out the FastStream Repo](https://github.com/ag2ai/faststream)

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
<img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support:** Integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Leverage Pydantic for efficient message serialization and validation, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Automatically generate comprehensive documentation, simplifying service integration and understanding.
*   **Intuitive Development Experience:** Benefit from full-typed editor support, catching errors early and streamlining development.
*   **Powerful Dependency Injection:** Manage service dependencies effectively with FastStream's built-in DI system.
*   **Testability:** Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Extend functionality with lifespans, custom serialization, and middleware integrations.
*   **Framework Compatibility:**  Seamlessly integrate with any HTTP framework, especially with the included FastAPI plugin.

---

## Getting Started

### Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-style operating systems. Install using `pip`:

```bash
pip install 'faststream[kafka]'  # Install with Kafka support
# or
pip install 'faststream[rabbit]' # Install with RabbitMQ support
# or
pip install 'faststream[nats]'  # Install with NATS support
# or
pip install 'faststream[redis]' # Install with Redis support
```

### Writing Application Code

Use the `@broker.subscriber` and `@broker.publisher` decorators to simplify consuming and producing data to/from event queues. FastStream leverages Pydantic for easy data serialization using type annotations.

**Example:**

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

**Using Pydantic Models:**

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

### Testing

Test your service using `TestBroker` within `pytest` for in-memory tests:

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

### Running the Application

Install the FastStream CLI:

```bash
pip install "faststream[cli]"
```

Then, run your application:

```bash
faststream run basic:app
```

Or use hot reload or multiprocessing scaling:

```bash
faststream run basic:app --reload # with hot reload
faststream run basic:app --workers 3 # with multiprocessing scaling
```

Learn more about FastStream CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates documentation according to the [AsyncAPI](https://www.asyncapi.com/) specification, visualizing the channels and message formats your application uses.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses a dependency management system similar to `pytest fixtures` and `FastAPI Depends`:

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

Use `MQBrokers` without a `FastStream` application, managing broker lifecycles:

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

Integrate with FastAPI using `KafkaRouter`:

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

Find more integration features [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay Connected

*   [GitHub Repository](https://github.com/ag2ai/faststream/)
*   [EN Discord server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

Thanks to all contributors who made the project better!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>