# FastStream: Build Scalable Microservices with Effortless Event Stream Integration

**Simplify your event-driven architecture with FastStream, the Python framework designed for easy and efficient message queue integration.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
<br/>
<br/>
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads/month](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
<br/>
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
<br/>
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
<img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
<br/>
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

FastStream makes building event-driven applications easier than ever, offering a robust set of features for modern microservices.

*   **Multiple Brokers:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization, ensuring data integrity.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically, simplifying service integration and documentation.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in the development cycle.
*   **Dependency Injection:** Efficiently manage service dependencies with FastStream's built-in DI system.
*   **Easy Testing:** Leverage in-memory testing for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Customize your application with extensions for lifespans, custom serialization, and middleware.
*   **Framework Compatibility:** Integrate FastStream with any HTTP framework, with special support for FastAPI.

Get started with FastStream and experience the simplicity and power of event-driven architectures.  [Learn more about FastStream](https://faststream.ag2.ai/latest/) or check out the source code on [GitHub](https://github.com/ag2ai/faststream).

---

## Installation

FastStream is compatible with Linux, macOS, Windows, and most Unix-style operating systems. Install it easily using pip:

```bash
pip install 'faststream[kafka]'  # For Kafka
pip install 'faststream[rabbit]' # For RabbitMQ
pip install 'faststream[nats]'   # For NATS
pip install 'faststream[redis]'  # For Redis
```

Note: FastStream uses PydanticV2 (written in Rust) by default, but will gracefully fallback to PydanticV1 if Rust support is unavailable.

---

## Writing App Code

FastStream simplifies event handling with easy-to-use decorators: `@broker.subscriber` for consumers and `@broker.publisher` for producers.  These decorators handle message parsing, networking, and documentation, letting you focus on your business logic.  Leverage Pydantic for type-safe input validation.

Example:

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

You can also use Pydantic's `BaseModel` for structured messages:

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

Test your FastStream services effortlessly with the `TestBroker` context manager. This allows you to run in-memory tests without a running broker.

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

Start your FastStream application using the command-line interface (CLI). First, install the CLI:

```bash
pip install "faststream[cli]"
```

Run your app with:

```bash
faststream run basic:app
```

Take advantage of hot-reloading and multiprocessing for enhanced development and scalability:

```bash
faststream run basic:app --reload  # Hot reload
faststream run basic:app --workers 3 # Multiprocessing
```

Learn more about FastStream CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates [AsyncAPI](https://www.asyncapi.com/) documentation for your project, helping you to understand your services' channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream's dependency management system leverages [FastDepends](https://lancetnik.github.io/FastDepends/), similar to pytest fixtures and FastAPI Depends.

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

## HTTP Frameworks Integrations

### Any Framework

Integrate FastStream's `MQBrokers` into any HTTP framework. Start and stop brokers within your application's lifespan.

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

Use FastStream within FastAPI applications. Import the `StreamRouter` and define message handlers using `@router.subscriber` and `@router.publisher`.

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

For more integration options, see the [FastAPI documentation](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/).

---

## Stay in Touch

*   Give our [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Thanks to all the amazing contributors who make this project possible!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>