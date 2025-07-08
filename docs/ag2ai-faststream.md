# FastStream: Build Microservices Effortlessly with Event Streams

**Simplify your event-driven architectures with FastStream, a powerful and intuitive framework for Python.**

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

*   **Multiple Broker Support:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Validate incoming messages with the power of Pydantic.
*   **Automatic AsyncAPI Docs:** Generate comprehensive documentation automatically.
*   **Intuitive Development:** Benefit from full-typed editor support for a smooth experience.
*   **Dependency Injection:** Efficiently manage service dependencies with a built-in DI system.
*   **Simplified Testing:** Utilize in-memory tests for a faster and more reliable CI/CD pipeline.
*   **Extensibility:** Leverage extensions for lifespans, custom serialization, and middleware.
*   **Framework Integration:** Easily integrate with any HTTP framework, especially FastAPI.

---

## Getting Started

FastStream simplifies the development of event-driven microservices by handling the complexities of message queue integration.  It automatically manages parsing, networking, and documentation, so you can focus on your core business logic.

### Installation

Install FastStream with your desired broker support using `pip`:

```bash
pip install 'faststream[kafka]'  # For Kafka
# or
pip install 'faststream[rabbit]' # For RabbitMQ
# or
pip install 'faststream[nats]'   # For NATS
# or
pip install 'faststream[redis]'  # For Redis
```

### Writing App Code

FastStream uses function decorators (`@broker.subscriber` and `@broker.publisher`) to simplify the process of consuming and producing data.  It also integrates Pydantic for easy data validation and serialization.

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

You can also use Pydantic models for structured data:

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

### Testing Your Service

FastStream provides `TestBroker` for easy in-memory testing without a running broker:

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

### Running Your Application

Run your application using the FastStream CLI:

1.  **Install the CLI:**  `pip install "faststream[cli]"`
2.  **Run the app:**  `faststream run basic:app`

Additional CLI features include hot-reloading (`--reload`) and multiprocessing scaling (`--workers`).

### Project Documentation

FastStream automatically generates documentation in the [AsyncAPI](https://www.asyncapi.com/) specification.

---

## Core Concepts

*   **Declarative Approach:** Define your message processing logic with simple decorators.
*   **Data Validation:** Pydantic integration ensures data integrity.
*   **Asynchronous Operations:** Built for high-performance event processing.
*   **Simplified Development:** Focus on business logic, not infrastructure.

---

## Dependencies

FastStream includes a dependency management system, similar to pytest fixtures and FastAPI Depends, using the [FastDepends](https://lancetnik.github.io/FastDepends/) library.

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
You can use FastStream's `MQBrokers` independently of a `FastStream` application by starting and stopping them within your application's lifespan.

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

FastStream seamlessly integrates with FastAPI using a StreamRouter:

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

## History

FastStream evolved from FastKafka and Propan, combining the best features of both to create a unified framework for streaming data processing.

---

## Stay Connected

*   **GitHub:** [Visit our Repository](https://github.com/ag2ai/faststream/)
*   **Discord:** [Join our Discord Server](https://discord.gg/qFm6aSqq59)
*   **Telegram:** [Join our Telegram Group](https://t.me/python_faststream)

---

## Contributors

Thanks to all the awesome contributors!  [![Contributors](https://contrib.rocks/image?repo=ag2ai/faststream)](https://github.com/ag2ai/faststream/graphs/contributors)

---

**Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)