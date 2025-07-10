# FastStream: Effortlessly Integrate Event Streams into Your Services

**Simplify microservice communication with FastStream, a powerful and user-friendly framework for building event-driven applications.**

---

[<img src="https://trendshift.io/api/badge/repositories/3043" alt="ag2ai%2Ffaststream | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/3043)
<br/>
<br/>

[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![Package version](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
<br/>
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
<br/>
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
<img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>
<a href="https://t.me/python_faststream" target="_blank">
  <img alt="Telegram" src="https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU"/>
</a>
<br/>
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support**: Seamlessly integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation**: Leverage Pydantic for robust data validation and serialization.
*   **Automatic Documentation**: Generate AsyncAPI documentation automatically for easy service discovery and integration.
*   **Developer-Friendly**: Enjoy full-typed editor support to catch errors early and improve developer experience.
*   **Powerful Dependency Injection**: Manage dependencies efficiently with FastStream's built-in DI system.
*   **Simplified Testing**: Utilize in-memory tests to speed up CI/CD pipelines and ensure code quality.
*   **Extensible Architecture**: Use extensions for lifespans, custom serialization and middleware.
*   **Framework Compatibility**: Fully compatible with any HTTP framework, including a dedicated plugin for FastAPI.

Read the full documentation at:  [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## How to Get Started

### Installation

FastStream is available for Linux, macOS, Windows and most Unix-style operating systems. Install it using pip:

```bash
pip install 'faststream[kafka]'   # Or use 'rabbit', 'nats', or 'redis'
```

By default **FastStream** uses **PydanticV2** written in **Rust**, but you can downgrade it manually, if your platform has no **Rust** support - **FastStream** will work correctly with **PydanticV1** as well.

### Writing App Code

FastStream simplifies message handling with `@broker.subscriber` and `@broker.publisher` decorators. It uses Pydantic for easy data serialization and validation, allowing you to focus on your core business logic.

Here's a basic example:

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

You can also use Pydantic models:

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

### Testing the Service

Use `TestBroker` for in-memory testing with pytest:

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

Install the CLI:

```bash
pip install "faststream[cli]"
```

Run your app:

```bash
faststream run basic:app  # Replace 'basic' with your module name
```

Enhance your development experience with hot reload and multiprocessing:

```bash
faststream run basic:app --reload
faststream run basic:app --workers 3
```

Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/)

---

## Project Documentation

FastStream automatically generates AsyncAPI documentation. This simplifies service integration by clearly showing channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses `FastDepends` for a dependency management system similar to `pytest fixtures` and `FastAPI Depends`.

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

## Integrations

### Any Framework

Use `MQBrokers` independently within your application's lifespan.

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

Integrate with FastAPI using a `StreamRouter`.

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

## Connect with the Community

*   **Star us on GitHub**: [https://github.com/ag2ai/faststream/](https://github.com/ag2ai/faststream/)
*   **Join our Discord**: [https://discord.gg/qFm6aSqq59](https://discord.gg/qFm6aSqq59)
*   **Join our Telegram**: [https://t.me/python_faststream](https://t.me/python_faststream)

---

## Contributors

Thank you to all the amazing contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>